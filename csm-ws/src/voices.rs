use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

/// Maps client-facing `voice_id` strings to numeric CSM speaker IDs.
/// Loaded from a JSON file at startup; falls back to a small built-in default.
#[derive(Debug, Clone)]
pub struct VoiceMap {
    inner: HashMap<String, u32>,
}

impl VoiceMap {
    pub fn default_map() -> Self {
        let mut inner = HashMap::new();
        // CSM-1B was trained with speakers 0..n. Provide a few stable aliases.
        inner.insert("default".into(), 0);
        inner.insert("speaker_0".into(), 0);
        inner.insert("speaker_1".into(), 1);
        inner.insert("speaker_2".into(), 2);
        inner.insert("speaker_3".into(), 3);
        Self { inner }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let inner: HashMap<String, u32> = serde_json::from_slice(&bytes)?;
        if inner.is_empty() {
            return Err(anyhow!("voice map file is empty"));
        }
        Ok(Self { inner })
    }

    pub fn resolve(&self, voice_id: &str) -> Option<u32> {
        // Direct lookup first.
        if let Some(&id) = self.inner.get(voice_id) {
            return Some(id);
        }
        // Allow numeric voice_ids to pass through unchanged ("0", "1", ...).
        voice_id.parse::<u32>().ok()
    }
}
