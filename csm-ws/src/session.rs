use crate::audio::{AudioEncoder, OutputFormat};
use crate::pool::CheckoutGuard;
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use candle_core::{DType, Tensor};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc;

/// Generation parameters captured from the first client message (or defaults).
#[derive(Debug, Clone)]
pub struct GenParams {
    pub speaker_id: u32,
    pub temperature: f64,
    pub top_k: usize,
    pub max_audio_len_ms: f32,
    pub buffer_size: usize,
    pub tokenizer_template: Option<String>,
}

impl GenParams {
    pub fn defaults(speaker_id: u32) -> Self {
        Self {
            speaker_id,
            temperature: 0.7,
            top_k: 100,
            max_audio_len_ms: 30_000.0,
            buffer_size: 4,
            tokenizer_template: None,
        }
    }

    fn apply(&mut self, msg: &ClientMessage) {
        if let Some(s) = msg.speaker_id {
            self.speaker_id = s;
        }
        if let Some(t) = msg.temperature {
            self.temperature = t;
        }
        if let Some(k) = msg.top_k {
            self.top_k = k;
        }
    }
}

pub struct Session {
    pub ws: WebSocket,
    pub guard: CheckoutGuard,
    pub params: GenParams,
    format: OutputFormat,
    text_buffer: String,
    encoder: AudioEncoder,
}

impl Session {
    pub fn new(
        ws: WebSocket,
        guard: CheckoutGuard,
        format: OutputFormat,
        params: GenParams,
    ) -> Result<Self> {
        let encoder = AudioEncoder::new(format)?;
        Ok(Self {
            ws,
            guard,
            params,
            format,
            text_buffer: String::new(),
            encoder,
        })
    }

    pub async fn run(mut self) -> Result<()> {
        // Read messages until EOS or socket closes. Generate when the buffer
        // hits a sentence boundary or the client requests a flush.
        loop {
            let next = self.ws.recv().await;
            let Some(frame) = next else { break };
            let frame = match frame {
                Ok(f) => f,
                Err(e) => {
                    log::warn!("ws recv error: {e}");
                    break;
                }
            };

            let text = match frame {
                Message::Text(t) => t,
                Message::Close(_) => break,
                Message::Ping(p) => {
                    self.ws.send(Message::Pong(p)).await.ok();
                    continue;
                }
                _ => continue,
            };

            let msg: ClientMessage = match serde_json::from_str(&text) {
                Ok(m) => m,
                Err(e) => {
                    let err = ServerMessage {
                        audio: None,
                        is_final: true,
                        alignment: None,
                        normalized_alignment: None,
                    };
                    log::warn!("invalid client message: {e}");
                    let _ = self.ws.send(Message::Text(serde_json::to_string(&err)?.into())).await;
                    break;
                }
            };

            self.params.apply(&msg);

            if msg.is_eos() {
                // Drain any leftover buffered text. If there's nothing buffered
                // (the common case, since sentence boundaries flush eagerly),
                // we don't emit another isFinal — the previous generation
                // already sent one.
                let chunk = std::mem::take(&mut self.text_buffer);
                if !chunk.trim().is_empty() {
                    self.generate_chunk(chunk).await?;
                }
                break;
            }

            if !msg.text.is_empty() {
                self.text_buffer.push_str(&msg.text);
            }

            let should_generate = msg.flush
                || msg.try_trigger_generation
                || sentence_boundary(&self.text_buffer);

            if should_generate {
                let chunk = std::mem::take(&mut self.text_buffer);
                if !chunk.trim().is_empty() {
                    self.generate_chunk(chunk).await?;
                }
            }
        }
        Ok(())
    }

    /// Run one generation pass for the given text and stream audio to the client.
    /// Always emits `isFinal: true` after the last audio chunk for this turn.
    async fn generate_chunk(&mut self, text: String) -> Result<()> {
        // Fresh encoder per turn so the resampler doesn't carry FFT state
        // across utterance boundaries (avoids subtle clicks at the seam).
        self.encoder = AudioEncoder::new(self.format)?;

        let params = self.params.clone();
        let generator = self.guard.generator.clone();

        // Bridge the blocking GPU loop to async. 8 = ~640ms of headroom @ 80ms/frame.
        let (frame_tx, frame_rx) = std_mpsc::sync_channel::<anyhow::Result<Tensor>>(8);
        // Async side: receive Vec<f32> samples ready for encoding.
        let (sample_tx, mut sample_rx) = mpsc::channel::<anyhow::Result<Vec<f32>>>(8);

        // Bridge thread: pull blocking std mpsc → push tokio mpsc. Also converts
        // tensor → Vec<f32> here so the async task only deals with Rust slices.
        let bridge = std::thread::spawn(move || {
            for item in frame_rx {
                let converted = item.and_then(tensor_to_f32_vec);
                if sample_tx.blocking_send(converted).is_err() {
                    break;
                }
            }
        });

        // Spawn the blocking generation. Holds the per-session generator mutex
        // for the duration; the pool guard gates concurrent sessions.
        let gen_handle = tokio::task::spawn_blocking(move || {
            let mut gen = generator.blocking_lock();
            gen.generate_to_channel(
                &text,
                params.speaker_id,
                params.max_audio_len_ms,
                params.temperature,
                params.top_k,
                params.buffer_size,
                params.tokenizer_template,
                frame_tx,
            );
        });

        // Pump audio chunks → encode → base64 → send.
        while let Some(item) = sample_rx.recv().await {
            let samples = match item {
                Ok(s) => s,
                Err(e) => {
                    log::error!("generation error: {e}");
                    break;
                }
            };
            let bytes = self.encoder.process(&samples)?;
            if !bytes.is_empty() {
                let msg = ServerMessage {
                    audio: Some(B64.encode(&bytes)),
                    is_final: false,
                    alignment: None,
                    normalized_alignment: None,
                };
                if self
                    .ws
                    .send(Message::Text(serde_json::to_string(&msg)?.into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }

        let _ = bridge.join();
        let _ = gen_handle.await;

        // Each generation is an independent CSM utterance (KV cache is cleared
        // at the start of generate_to_channel). Flush any encoder tail and
        // send isFinal so the client knows this turn is complete.
        let tail = self.encoder.finish().unwrap_or_default();
        self.send_final(tail).await?;
        Ok(())
    }

    async fn send_final(&mut self, tail: Vec<u8>) -> Result<()> {
        let audio = if tail.is_empty() {
            None
        } else {
            Some(B64.encode(&tail))
        };
        let msg = ServerMessage {
            audio,
            is_final: true,
            alignment: None,
            normalized_alignment: None,
        };
        let _ = self
            .ws
            .send(Message::Text(serde_json::to_string(&msg)?.into()))
            .await;
        Ok(())
    }
}

fn tensor_to_f32_vec(t: Tensor) -> anyhow::Result<Vec<f32>> {
    let v = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v)
}

/// Returns true if the buffer ends with sentence-terminating punctuation
/// followed by whitespace, OR is long enough that we should flush regardless.
/// This matches ElevenLabs' default "chunk_length_schedule" behavior loosely.
fn sentence_boundary(buf: &str) -> bool {
    if buf.len() >= 280 {
        return true;
    }
    let trimmed = buf.trim_end();
    matches!(trimmed.chars().last(), Some('.') | Some('!') | Some('?') | Some(';'))
        && buf.ends_with(' ')
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundary_sentence() {
        assert!(sentence_boundary("Hello world. "));
        assert!(sentence_boundary("Are you there? "));
        assert!(!sentence_boundary("Hello world"));
        assert!(!sentence_boundary("Hello world."));
    }
    #[test]
    fn boundary_long() {
        let s = "a ".repeat(200);
        assert!(sentence_boundary(&s));
    }
}
