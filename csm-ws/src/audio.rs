use anyhow::{anyhow, Result};
use mp3lame_encoder::{Builder as Mp3Builder, FlushNoGap, MonoPcm};
use rubato::{FftFixedInOut, Resampler};

pub const NATIVE_SAMPLE_RATE: usize = 24_000;

/// All output formats the server can produce.
/// Naming matches ElevenLabs' `output_format` query param.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Pcm16000,
    Pcm22050,
    Pcm24000,
    Pcm44100,
    Ulaw8000,
    Wav16000,
    Wav22050,
    Wav24000,
    Wav44100,
    /// MP3 at sample_rate / kbps.
    Mp3(u32, u32),
}

impl OutputFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pcm_16000" => Some(Self::Pcm16000),
            "pcm_22050" => Some(Self::Pcm22050),
            "pcm_24000" => Some(Self::Pcm24000),
            "pcm_44100" => Some(Self::Pcm44100),
            "ulaw_8000" => Some(Self::Ulaw8000),
            "wav_16000" => Some(Self::Wav16000),
            "wav_22050" => Some(Self::Wav22050),
            "wav_24000" => Some(Self::Wav24000),
            "wav_44100" => Some(Self::Wav44100),
            "mp3_22050_32" => Some(Self::Mp3(22_050, 32)),
            "mp3_44100_32" => Some(Self::Mp3(44_100, 32)),
            "mp3_44100_64" => Some(Self::Mp3(44_100, 64)),
            "mp3_44100_96" => Some(Self::Mp3(44_100, 96)),
            "mp3_44100_128" => Some(Self::Mp3(44_100, 128)),
            "mp3_44100_192" => Some(Self::Mp3(44_100, 192)),
            _ => None,
        }
    }

    pub fn target_rate(&self) -> usize {
        match self {
            Self::Pcm16000 | Self::Wav16000 => 16_000,
            Self::Pcm22050 | Self::Wav22050 => 22_050,
            Self::Pcm24000 | Self::Wav24000 => 24_000,
            Self::Pcm44100 | Self::Wav44100 => 44_100,
            Self::Ulaw8000 => 8_000,
            Self::Mp3(rate, _) => *rate as usize,
        }
    }

    /// HTTP `Content-Type` for this format. Used by the REST endpoints.
    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Pcm16000 | Self::Pcm22050 | Self::Pcm24000 | Self::Pcm44100 => {
                "application/octet-stream"
            }
            Self::Ulaw8000 => "audio/basic",
            Self::Wav16000 | Self::Wav22050 | Self::Wav24000 | Self::Wav44100 => "audio/wav",
            Self::Mp3(_, _) => "audio/mpeg",
        }
    }
}

enum Codec {
    /// Raw little-endian s16 PCM. Used for both `pcm_*` and the body of `wav_*`.
    Pcm,
    Ulaw,
    /// LAME MP3 encoder. Stateful — output bytes can lag input.
    Mp3 { encoder: mp3lame_encoder::Encoder },
}

pub struct AudioEncoder {
    format: OutputFormat,
    resampler: Option<FftFixedInOut<f32>>,
    leftover: Vec<f32>,
    chunk_in: usize,
    codec: Codec,
    /// True until the first `process()` call has emitted a container header.
    needs_header: bool,
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

        let codec = match format {
            OutputFormat::Ulaw8000 => Codec::Ulaw,
            OutputFormat::Mp3(rate, kbps) => {
                let mut b = Mp3Builder::new().ok_or_else(|| anyhow!("LAME init failed"))?;
                b.set_num_channels(1).map_err(|e| anyhow!("{e:?}"))?;
                b.set_sample_rate(rate).map_err(|e| anyhow!("{e:?}"))?;
                let br = bitrate_from_kbps(kbps)?;
                b.set_brate(br).map_err(|e| anyhow!("{e:?}"))?;
                b.set_quality(mp3lame_encoder::Quality::Best)
                    .map_err(|e| anyhow!("{e:?}"))?;
                let encoder = b.build().map_err(|e| anyhow!("{e:?}"))?;
                Codec::Mp3 { encoder }
            }
            _ => Codec::Pcm,
        };

        // Header is emitted only by the WAV variants on the first call.
        let needs_header = matches!(
            format,
            OutputFormat::Wav16000
                | OutputFormat::Wav22050
                | OutputFormat::Wav24000
                | OutputFormat::Wav44100
        );

        Ok(Self {
            format,
            resampler,
            leftover: Vec::new(),
            chunk_in,
            codec,
            needs_header,
        })
    }

    /// Process a chunk of native-rate (24kHz) f32 mono samples in [-1, 1].
    /// Returns encoded bytes ready for the wire (including any container header
    /// on the first call).
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let resampled = self.resample(samples)?;
        let mut out = Vec::new();
        if self.needs_header {
            out.extend_from_slice(&streaming_wav_header(self.format.target_rate() as u32, 1, 16));
            self.needs_header = false;
        }
        out.extend_from_slice(&self.encode(&resampled)?);
        Ok(out)
    }

    /// Flush any buffered samples and finalize the codec. Returns trailing bytes.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        let resampled = self.flush_resampler()?;
        let mut out = Vec::new();
        if self.needs_header {
            // Empty stream — still write a valid header so the file plays.
            out.extend_from_slice(&streaming_wav_header(self.format.target_rate() as u32, 1, 16));
            self.needs_header = false;
        }
        out.extend_from_slice(&self.encode(&resampled)?);
        out.extend_from_slice(&self.flush_codec()?);
        Ok(out)
    }

    fn resample(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        match self.resampler.as_mut() {
            None => Ok(samples.to_vec()),
            Some(r) => {
                self.leftover.extend_from_slice(samples);
                let mut out: Vec<f32> = Vec::new();
                while self.leftover.len() >= self.chunk_in {
                    let block: Vec<f32> = self.leftover.drain(..self.chunk_in).collect();
                    let processed = r.process(&[block], None)?;
                    out.extend_from_slice(&processed[0]);
                }
                Ok(out)
            }
        }
    }

    fn flush_resampler(&mut self) -> Result<Vec<f32>> {
        match self.resampler.as_mut() {
            None => Ok(Vec::new()),
            Some(r) => {
                if self.leftover.is_empty() {
                    return Ok(Vec::new());
                }
                let mut block = std::mem::take(&mut self.leftover);
                block.resize(self.chunk_in, 0.0);
                let processed = r.process(&[block], None)?;
                Ok(processed.into_iter().next().unwrap_or_default())
            }
        }
    }

    fn encode(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        Ok(match &mut self.codec {
            Codec::Ulaw => samples.iter().map(|&s| linear_to_ulaw(s)).collect(),
            Codec::Pcm => f32_to_s16_le(samples),
            Codec::Mp3 { encoder } => {
                let pcm: Vec<i16> = samples
                    .iter()
                    .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                    .collect();
                let mut buf = vec![std::mem::MaybeUninit::uninit(); mp3lame_encoder::max_required_buffer_size(pcm.len())];
                let n = encoder
                    .encode(MonoPcm(&pcm), buf.as_mut_slice())
                    .map_err(|e| anyhow!("mp3 encode: {e:?}"))?;
                unsafe { mp3_assume_init(buf, n) }
            }
        })
    }

    fn flush_codec(&mut self) -> Result<Vec<u8>> {
        Ok(match &mut self.codec {
            Codec::Mp3 { encoder } => {
                let mut buf = vec![std::mem::MaybeUninit::uninit(); mp3lame_encoder::max_required_buffer_size(0)];
                let n = encoder
                    .flush::<FlushNoGap>(buf.as_mut_slice())
                    .map_err(|e| anyhow!("mp3 flush: {e:?}"))?;
                unsafe { mp3_assume_init(buf, n) }
            }
            _ => Vec::new(),
        })
    }
}

unsafe fn mp3_assume_init(buf: Vec<std::mem::MaybeUninit<u8>>, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let ptr = buf.as_ptr() as *const u8;
    out.extend_from_slice(std::slice::from_raw_parts(ptr, n));
    out
}

fn f32_to_s16_le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bitrate_from_kbps(kbps: u32) -> Result<mp3lame_encoder::Bitrate> {
    use mp3lame_encoder::Bitrate::*;
    Ok(match kbps {
        32 => Kbps32,
        64 => Kbps64,
        96 => Kbps96,
        128 => Kbps128,
        160 => Kbps160,
        192 => Kbps192,
        256 => Kbps256,
        320 => Kbps320,
        _ => return Err(anyhow!("unsupported mp3 kbps {kbps}")),
    })
}

/// 44-byte WAV header for streaming use. The RIFF size and data-chunk size
/// fields are set to `0xFFFFFFFF` since we don't know the final length up
/// front; ffplay/ffmpeg/browser players all handle this.
fn streaming_wav_header(sample_rate: u32, num_channels: u16, bits_per_sample: u16) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&num_channels.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    h.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * bits_per_sample / 8;
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits_per_sample.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h
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
        assert_eq!(linear_to_ulaw(0.0), 0xFF);
    }

    #[test]
    fn ulaw_extremes_in_range() {
        let _ = linear_to_ulaw(1.0);
        let _ = linear_to_ulaw(-1.0);
    }

    #[test]
    fn pcm_passthrough_24k() {
        let mut enc = AudioEncoder::new(OutputFormat::Pcm24000).unwrap();
        let bytes = enc.process(&[0.0; 480]).unwrap();
        assert_eq!(bytes.len(), 480 * 2);
    }

    #[test]
    fn ulaw_8k_resample_roughly_third() {
        let mut enc = AudioEncoder::new(OutputFormat::Ulaw8000).unwrap();
        let mut total = 0;
        for _ in 0..10 {
            total += enc.process(&[0.0; 480]).unwrap().len();
        }
        total += enc.finish().unwrap().len();
        assert!((1500..=1700).contains(&total), "got {total}");
    }

    #[test]
    fn wav_emits_header_then_pcm() {
        let mut enc = AudioEncoder::new(OutputFormat::Wav24000).unwrap();
        let bytes = enc.process(&[0.0; 480]).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // 44 header + 480*2 PCM body.
        assert_eq!(bytes.len(), 44 + 480 * 2);
        // Subsequent calls don't re-emit the header.
        let more = enc.process(&[0.0; 480]).unwrap();
        assert_eq!(more.len(), 480 * 2);
    }

    #[test]
    fn mp3_produces_some_bytes() {
        let mut enc = AudioEncoder::new(OutputFormat::Mp3(44_100, 128)).unwrap();
        // Feed a couple of seconds of silence so MP3 emits at least one frame.
        let mut total = 0;
        for _ in 0..200 {
            total += enc.process(&[0.0; 480]).unwrap().len();
        }
        total += enc.finish().unwrap().len();
        assert!(total > 100, "expected mp3 output, got {total} bytes");
    }
}
