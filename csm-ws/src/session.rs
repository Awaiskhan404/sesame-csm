use crate::audio::OutputFormat;
use crate::generation::{self, GenParams};
use crate::pool::CheckoutGuard;
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use base64::{engine::general_purpose::STANDARD as B64, Engine};

pub struct Session {
    ws: WebSocket,
    guard: CheckoutGuard,
    format: OutputFormat,
    params: GenParams,
    text_buffer: String,
}

impl Session {
    pub fn new(ws: WebSocket, guard: CheckoutGuard, format: OutputFormat, params: GenParams) -> Self {
        Self {
            ws,
            guard,
            format,
            params,
            text_buffer: String::new(),
        }
    }

    pub async fn run(mut self) -> Result<()> {
        loop {
            let Some(frame) = self.ws.recv().await else { break };
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
                    log::warn!("invalid client message: {e}");
                    let _ = self
                        .ws
                        .send(Message::Text(serde_json::to_string(&ServerMessage {
                            audio: None,
                            is_final: true,
                            alignment: None,
                            normalized_alignment: None,
                        })?.into()))
                        .await;
                    break;
                }
            };

            apply_overrides(&mut self.params, &msg);

            if msg.is_eos() {
                // Drain any leftover buffered text. If nothing is buffered
                // (the common case, since sentence boundaries flush eagerly),
                // skip — the previous turn already sent isFinal.
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

    async fn generate_chunk(&mut self, text: String) -> Result<()> {
        let mut rx = generation::run(
            self.guard.generator.clone(),
            text,
            self.format,
            self.params.clone(),
        )?;
        while let Some(item) = rx.recv().await {
            let bytes = match item {
                Ok(b) => b,
                Err(e) => {
                    log::error!("generation error: {e}");
                    break;
                }
            };
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
                return Ok(());
            }
        }
        // End-of-turn marker.
        let final_msg = ServerMessage {
            audio: None,
            is_final: true,
            alignment: None,
            normalized_alignment: None,
        };
        let _ = self
            .ws
            .send(Message::Text(serde_json::to_string(&final_msg)?.into()))
            .await;
        Ok(())
    }
}

fn apply_overrides(params: &mut GenParams, msg: &ClientMessage) {
    if let Some(s) = msg.speaker_id {
        params.speaker_id = s;
    }
    if let Some(t) = msg.temperature {
        params.temperature = t;
    }
    if let Some(k) = msg.top_k {
        params.top_k = k;
    }
}

/// Returns true if the buffer ends with sentence-terminating punctuation
/// followed by whitespace, OR is long enough that we should flush regardless.
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
