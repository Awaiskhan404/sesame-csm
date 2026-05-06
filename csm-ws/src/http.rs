//! HTTP TTS endpoints. Mirrors the ElevenLabs REST API shape:
//!   POST /v1/text-to-speech/{voice_id}                          → full body
//!   POST /v1/text-to-speech/{voice_id}/stream                    → chunked
//!   POST /v1/text-to-speech/{voice_id}/stream/with-timestamps    → SSE
//!
//! The generation core is shared with the WebSocket handler via
//! `crate::generation::run`.

use crate::audio::OutputFormat;
use crate::generation::{self, GenParams};
use crate::AppState;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use futures_util::{Stream, StreamExt};
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Deserialize, Debug, Default)]
pub struct TtsQuery {
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(rename = "xi-api-key", default)]
    pub xi_api_key: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)] // ElevenLabs-parity fields are accepted but not all are acted on.
pub struct TtsRequest {
    pub text: String,

    // Generation knobs (Sesame-specific, optional).
    #[serde(default)]
    pub speaker_id: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_audio_len_ms: Option<f32>,

    // Accepted for ElevenLabs parity but ignored:
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub voice_settings: Option<serde_json::Value>,
    #[serde(default)]
    pub previous_text: Option<String>,
    #[serde(default)]
    pub next_text: Option<String>,
}

fn check_auth(state: &AppState, headers: &HeaderMap, q: &TtsQuery) -> Result<(), Response> {
    if let Some(expected) = &state.api_key {
        let provided = q
            .xi_api_key
            .as_deref()
            .or_else(|| headers.get("xi-api-key").and_then(|v| v.to_str().ok()));
        if provided != Some(expected.as_str()) {
            return Err((StatusCode::UNAUTHORIZED, "invalid api key").into_response());
        }
    }
    Ok(())
}

fn build_params(state: &AppState, voice_id: &str, body: &TtsRequest) -> Result<GenParams, Response> {
    let speaker_id = state
        .voices
        .resolve(voice_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("unknown voice_id '{voice_id}'")).into_response())?;
    let mut params = GenParams::defaults(speaker_id);
    // REST default cap is shorter than WS — short utterances shouldn't pay
    // for runaway generations when CSM forgets to emit end-of-generation.
    params.max_audio_len_ms = state.rest_max_audio_len_ms;
    if let Some(s) = body.speaker_id {
        params.speaker_id = s;
    }
    if let Some(t) = body.temperature {
        params.temperature = t;
    }
    if let Some(k) = body.top_k {
        params.top_k = k;
    }
    if let Some(m) = body.max_audio_len_ms {
        params.max_audio_len_ms = m;
    }
    Ok(params)
}

fn pick_format(q: &TtsQuery) -> OutputFormat {
    q.output_format
        .as_deref()
        .and_then(OutputFormat::parse)
        // ElevenLabs default is mp3_44100_128.
        .unwrap_or(OutputFormat::Mp3(44_100, 128))
}

/// `POST /v1/text-to-speech/{voice_id}` — full body, accumulates the entire
/// generation before responding. Returns the same content-type as the
/// streaming variant (audio/mpeg, audio/wav, etc.) but with `Content-Length`.
pub async fn tts_full(
    Path(voice_id): Path<String>,
    Query(q): Query<TtsQuery>,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsRequest>,
) -> Response {
    if let Err(r) = check_auth(&state, &headers, &q) {
        return r;
    }
    let format = pick_format(&q);
    let params = match build_params(&state, &voice_id, &body) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let guard = state.pool.checkout().await;
    let mut rx = match generation::run(guard.generator.clone(), body.text, format, params) {
        Ok(rx) => rx,
        Err(e) => return error_response(e),
    };

    let mut buf: Vec<u8> = Vec::new();
    while let Some(item) = rx.recv().await {
        match item {
            Ok(b) => buf.extend_from_slice(&b),
            Err(e) => return error_response(e),
        }
    }
    drop(guard);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type())
        .header(header::CONTENT_LENGTH, buf.len())
        .body(Body::from(buf))
        .unwrap()
}

/// `POST /v1/text-to-speech/{voice_id}/stream` — HTTP chunked transfer of the
/// raw audio bytes (in the requested `output_format`). For WAV/MP3 the bytes
/// are a valid streamable container.
pub async fn tts_stream(
    Path(voice_id): Path<String>,
    Query(q): Query<TtsQuery>,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsRequest>,
) -> Response {
    if let Err(r) = check_auth(&state, &headers, &q) {
        return r;
    }
    let format = pick_format(&q);
    let params = match build_params(&state, &voice_id, &body) {
        Ok(p) => p,
        Err(r) => return r,
    };

    let guard = state.pool.checkout().await;
    let rx = match generation::run(guard.generator.clone(), body.text, format, params) {
        Ok(rx) => rx,
        Err(e) => return error_response(e),
    };

    // Wrap the receiver in a stream that holds onto `guard` so the pool slot
    // isn't released until the client has finished reading.
    let stream = ReceiverGuardStream::new(rx, guard).map(|item| {
        item.map(axum::body::Bytes::from)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type())
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// `POST /v1/text-to-speech/{voice_id}/stream/with-timestamps` — SSE.
/// Each event is JSON with `audio_base64`, `alignment`, and `normalized_alignment`.
/// CSM-1B doesn't expose forced-alignment data, so the alignment fields are
/// always `null` (clients depending on them will need to handle that).
pub async fn tts_stream_timestamps(
    Path(voice_id): Path<String>,
    Query(q): Query<TtsQuery>,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
    Json(body): Json<TtsRequest>,
) -> Response {
    if let Err(r) = check_auth(&state, &headers, &q) {
        return r;
    }
    let format = pick_format(&q);
    let params = match build_params(&state, &voice_id, &body) {
        Ok(p) => p,
        Err(r) => return r,
    };

    let guard = state.pool.checkout().await;
    let rx = match generation::run(guard.generator.clone(), body.text, format, params) {
        Ok(rx) => rx,
        Err(e) => return error_response(e),
    };

    let stream = ReceiverGuardStream::new(rx, guard).map(|item| -> Result<Event, Infallible> {
        let event = match item {
            Ok(b) => {
                let payload = serde_json::json!({
                    "audio_base64": B64.encode(&b),
                    "alignment": serde_json::Value::Null,
                    "normalized_alignment": serde_json::Value::Null,
                });
                Event::default().data(payload.to_string())
            }
            Err(e) => Event::default()
                .event("error")
                .data(serde_json::json!({ "error": e.to_string() }).to_string()),
        };
        Ok(event)
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn error_response(e: anyhow::Error) -> Response {
    log::error!("tts handler error: {e}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}

/// Wraps a `Receiver<Result<Vec<u8>>>` so the inner pool checkout stays alive
/// until the stream is dropped (i.e. until the client finishes reading or
/// disconnects).
struct ReceiverGuardStream<T> {
    inner: ReceiverStream<Result<T, anyhow::Error>>,
    _guard: crate::pool::CheckoutGuard,
}

impl<T> ReceiverGuardStream<T> {
    fn new(
        rx: tokio::sync::mpsc::Receiver<Result<T, anyhow::Error>>,
        guard: crate::pool::CheckoutGuard,
    ) -> Self {
        Self {
            inner: ReceiverStream::new(rx),
            _guard: guard,
        }
    }
}

impl<T> Stream for ReceiverGuardStream<T> {
    type Item = Result<T, anyhow::Error>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}
