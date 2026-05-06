#![allow(dead_code)]
// Some fields are accepted purely for ElevenLabs protocol parity (so client
// libraries don't error on them) but are not acted upon.
use serde::{Deserialize, Serialize};

/// Initial client message containing generation parameters.
/// ElevenLabs sends these on the first frame; we accept them on any frame
/// (only honored on the first one per session).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct VoiceSettings {
    pub stability: Option<f32>,
    pub similarity_boost: Option<f32>,
    pub style: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerationConfig {
    /// Server-side text accumulation. Server flushes when it sees these chars.
    /// We use a sentence-boundary set by default.
    #[serde(default)]
    pub chunk_length_schedule: Option<Vec<u32>>,
}

/// A message from the client → server over the WebSocket.
///
/// ElevenLabs semantics:
///   - `text` may be a partial fragment ending in a space.
///   - `text == ""` signals end-of-stream (close).
///   - `flush == true` forces the server to generate whatever is buffered.
#[derive(Debug, Clone, Deserialize)]
pub struct ClientMessage {
    pub text: String,
    #[serde(default)]
    pub flush: bool,
    #[serde(default)]
    pub try_trigger_generation: bool,
    #[serde(default)]
    pub voice_settings: Option<VoiceSettings>,
    #[serde(default)]
    pub generation_config: Option<GenerationConfig>,
    #[serde(default)]
    pub xi_api_key: Option<String>,
    /// Sesame-specific override (not part of ElevenLabs API).
    #[serde(default)]
    pub speaker_id: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

impl ClientMessage {
    pub fn is_eos(&self) -> bool {
        self.text.is_empty() && !self.flush && !self.try_trigger_generation
    }
}

/// A message from the server → client. Mirrors the ElevenLabs response shape.
#[derive(Debug, Clone, Serialize)]
pub struct ServerMessage {
    /// base64 of the encoded audio chunk in the requested output format.
    /// `None` on the final message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    #[serde(rename = "isFinal")]
    pub is_final: bool,
    /// Character-level alignment. We don't compute this for CSM; sent as null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alignment: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_alignment: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorMessage {
    pub error: String,
    pub code: u16,
    pub message: String,
}
