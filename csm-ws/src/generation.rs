use crate::audio::{AudioEncoder, OutputFormat};
use anyhow::Result;
use candle_core::{DType, Tensor};
use csm_rs::Generator;
use std::sync::{mpsc as std_mpsc, Arc};
use tokio::sync::{mpsc, Mutex};

/// Generation parameters captured from request input or query/CLI defaults.
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
}

/// Run one generation pass against the given (already checked-out) generator.
/// Returns a tokio mpsc receiver that yields encoded audio chunks in the
/// requested `format` as they're produced, plus a final tail flush. The
/// receiver closes when generation ends.
///
/// The caller is responsible for keeping the pool checkout alive (or dropping
/// it when this stream is fully consumed).
pub fn run(
    generator: Arc<Mutex<Generator>>,
    text: String,
    format: OutputFormat,
    params: GenParams,
) -> Result<mpsc::Receiver<Result<Vec<u8>>>> {
    let mut encoder = AudioEncoder::new(format)?;
    let (out_tx, out_rx) = mpsc::channel::<Result<Vec<u8>>>(8);

    // Bridge between the blocking GPU loop (std mpsc) and async (tokio mpsc).
    let (frame_tx, frame_rx) = std_mpsc::sync_channel::<anyhow::Result<Tensor>>(8);
    let (sample_tx, mut sample_rx) = mpsc::channel::<anyhow::Result<Vec<f32>>>(8);

    // Tensor → Vec<f32> bridge thread.
    std::thread::spawn(move || {
        for item in frame_rx {
            let converted = item.and_then(tensor_to_f32_vec);
            if sample_tx.blocking_send(converted).is_err() {
                break;
            }
        }
    });

    // Blocking generation thread. Holds the generator mutex for its lifetime.
    let gp = params.clone();
    tokio::task::spawn_blocking(move || {
        let mut gen = generator.blocking_lock();
        gen.generate_to_channel(
            &text,
            gp.speaker_id,
            gp.max_audio_len_ms,
            gp.temperature,
            gp.top_k,
            gp.buffer_size,
            gp.tokenizer_template,
            frame_tx,
        );
    });

    // Async pump: encode + forward.
    tokio::spawn(async move {
        while let Some(item) = sample_rx.recv().await {
            let samples = match item {
                Ok(s) => s,
                Err(e) => {
                    let _ = out_tx.send(Err(e)).await;
                    return;
                }
            };
            match encoder.process(&samples) {
                Ok(b) if b.is_empty() => {}
                Ok(b) => {
                    if out_tx.send(Ok(b)).await.is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = out_tx.send(Err(e)).await;
                    return;
                }
            }
        }
        // Final encoder flush (codec tail bytes + any leftover resampled audio).
        match encoder.finish() {
            Ok(b) if !b.is_empty() => {
                let _ = out_tx.send(Ok(b)).await;
            }
            Ok(_) => {}
            Err(e) => {
                let _ = out_tx.send(Err(e)).await;
            }
        }
    });

    Ok(out_rx)
}

fn tensor_to_f32_vec(t: Tensor) -> anyhow::Result<Vec<f32>> {
    let v = t.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v)
}
