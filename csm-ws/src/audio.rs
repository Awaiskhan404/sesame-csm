use anyhow::Result;
use rubato::{FftFixedInOut, Resampler};
use serde::Deserialize;

pub const NATIVE_SAMPLE_RATE: usize = 24_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Pcm16000,
    Pcm22050,
    Pcm24000,
    Pcm44100,
    Ulaw8000,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pcm_16000" => Some(Self::Pcm16000),
            "pcm_22050" => Some(Self::Pcm22050),
            "pcm_24000" => Some(Self::Pcm24000),
            "pcm_44100" => Some(Self::Pcm44100),
            "ulaw_8000" => Some(Self::Ulaw8000),
            _ => None,
        }
    }

    pub fn target_rate(&self) -> usize {
        match self {
            Self::Pcm16000 => 16_000,
            Self::Pcm22050 => 22_050,
            Self::Pcm24000 => 24_000,
            Self::Pcm44100 => 44_100,
            Self::Ulaw8000 => 8_000,
        }
    }
}

pub struct AudioEncoder {
    format: OutputFormat,
    resampler: Option<FftFixedInOut<f32>>,
    leftover: Vec<f32>,
    chunk_in: usize,
}

impl AudioEncoder {
    pub fn new(format: OutputFormat) -> Result<Self> {
        let target = format.target_rate();
        let resampler = if target == NATIVE_SAMPLE_RATE {
            None
        } else {
            // 480 input frames @ 24kHz = 20ms — small block keeps streaming latency low.
            Some(FftFixedInOut::<f32>::new(NATIVE_SAMPLE_RATE, target, 480, 1)?)
        };
        let chunk_in = resampler.as_ref().map(|r| r.input_frames_next()).unwrap_or(0);
        Ok(Self {
            format,
            resampler,
            leftover: Vec::new(),
            chunk_in,
        })
    }

    /// Process a chunk of native-rate (24kHz) f32 mono samples in [-1, 1].
    /// Returns encoded bytes ready for the wire.
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let resampled = match self.resampler.as_mut() {
            None => samples.to_vec(),
            Some(r) => {
                self.leftover.extend_from_slice(samples);
                let mut out: Vec<f32> = Vec::new();
                while self.leftover.len() >= self.chunk_in {
                    let block: Vec<f32> = self.leftover.drain(..self.chunk_in).collect();
                    let processed = r.process(&[block], None)?;
                    out.extend_from_slice(&processed[0]);
                }
                out
            }
        };
        Ok(self.encode(&resampled))
    }

    /// Flush any remaining buffered samples by zero-padding to a full block.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        let resampled = match self.resampler.as_mut() {
            None => Vec::new(),
            Some(r) => {
                if self.leftover.is_empty() {
                    Vec::new()
                } else {
                    let mut block = std::mem::take(&mut self.leftover);
                    block.resize(self.chunk_in, 0.0);
                    let processed = r.process(&[block], None)?;
                    processed.into_iter().next().unwrap_or_default()
                }
            }
        };
        Ok(self.encode(&resampled))
    }

    fn encode(&self, samples: &[f32]) -> Vec<u8> {
        match self.format {
            OutputFormat::Ulaw8000 => samples.iter().map(|&s| linear_to_ulaw(s)).collect(),
            _ => {
                let mut out = Vec::with_capacity(samples.len() * 2);
                for &s in samples {
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
        }
    }
}

/// ITU-T G.711 µ-law encoding (μ=255). Matches Twilio's expected format.
fn linear_to_ulaw(sample: f32) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;

    let s = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i32;
    let sign: u8 = if s < 0 { 0x80 } else { 0x00 };
    let mut mag = s.abs().min(CLIP) + BIAS;

    let mut exponent: u8 = 7;
    let mask = 0x4000;
    while exponent > 0 && (mag & mask) == 0 {
        exponent -= 1;
        mag <<= 1;
    }
    let mantissa = ((mag >> (exponent as i32 + 3)) & 0x0F) as u8;
    !(sign | (exponent << 4) | mantissa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulaw_silence_is_0xff() {
        // µ-law encodes silence as 0xFF.
        assert_eq!(linear_to_ulaw(0.0), 0xFF);
    }

    #[test]
    fn ulaw_extremes_in_range() {
        let _ = linear_to_ulaw(1.0);
        let _ = linear_to_ulaw(-1.0);
    }

    #[test]
    fn passthrough_24k() {
        let mut enc = AudioEncoder::new(OutputFormat::Pcm24000).unwrap();
        let bytes = enc.process(&[0.0; 480]).unwrap();
        assert_eq!(bytes.len(), 480 * 2);
    }

    #[test]
    fn ulaw_8k_resample_roughly_third() {
        let mut enc = AudioEncoder::new(OutputFormat::Ulaw8000).unwrap();
        let mut total = 0;
        // Feed 4800 samples (200ms @ 24k) — expect ~1600 µ-law bytes (200ms @ 8k).
        for _ in 0..10 {
            total += enc.process(&[0.0; 480]).unwrap().len();
        }
        total += enc.finish().unwrap().len();
        assert!((1500..=1700).contains(&total), "got {total}");
    }
}
