mod audio;
mod pool;
mod protocol;
mod session;
mod voices;

use anyhow::Result;
use audio::OutputFormat;
use axum::{
    extract::{ws::WebSocketUpgrade, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use clap::Parser;
use csm_rs::GeneratorArgs;
use pool::GeneratorPool;
use serde::Deserialize;
use session::{GenParams, Session};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use voices::VoiceMap;

#[derive(Parser, Debug)]
#[command(author, version, about = "Sesame CSM-1B WebSocket TTS server")]
struct Args {
    #[arg(long, default_value = "0.0.0.0", env = "HOST")]
    host: String,
    #[arg(long, default_value_t = 8080, env = "PORT")]
    port: u16,
    #[arg(long, env = "API_KEY", help = "If set, clients must pass ?xi-api-key=<key> or 'xi-api-key' header.")]
    api_key: Option<String>,

    // ----- model loading (mirrors csm-core) -----
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, help = "Data type for model weights: f32, f16, bf16. Defaults to f16 on CUDA, f32 on CPU.")]
    dtype: Option<String>,
    #[arg(long)]
    weights_path: Option<PathBuf>,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    model_path: Option<PathBuf>,
    #[arg(long)]
    model_file: Option<String>,
    #[arg(long)]
    index_file: Option<String>,
    #[arg(long)]
    tokenizer_id: Option<String>,

    // ----- runtime knobs -----
    #[arg(long, default_value_t = 1, env = "POOL_SIZE", help = "Number of generator instances (= max concurrent sessions).")]
    pool_size: usize,
    #[arg(long, env = "VOICES_FILE", help = "JSON file mapping voice_id strings to numeric speaker ids.")]
    voices_file: Option<PathBuf>,
}

struct AppState {
    pool: GeneratorPool,
    voices: VoiceMap,
    api_key: Option<String>,
}

fn parse_dtype(s: &str) -> Result<candle_core::DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(candle_core::DType::F32),
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        _ => anyhow::bail!("Unsupported dtype '{}'. Use f32, f16, or bf16.", s),
    }
}

fn pick_device(cpu: bool) -> candle_core::Device {
    if cpu {
        return candle_core::Device::Cpu;
    }
    if let Ok(d) = candle_core::Device::new_cuda(0) {
        return d;
    }
    if let Ok(d) = candle_core::Device::new_metal(0) {
        return d;
    }
    candle_core::Device::Cpu
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let device = pick_device(args.cpu);
    let dtype = args.dtype.as_deref().map(parse_dtype).transpose()?;
    log::info!("Using device {:?}, dtype {:?}", device, dtype);

    let gen_args = GeneratorArgs {
        weights_path: args.weights_path,
        model_id: args.model_id,
        model_path: args.model_path,
        model_file: args.model_file,
        index_file: args.index_file,
        tokenizer_id: args.tokenizer_id,
        device,
        dtype,
    };

    let pool = GeneratorPool::new(args.pool_size, gen_args).await?;
    let voices = match &args.voices_file {
        Some(p) => VoiceMap::from_file(p)?,
        None => VoiceMap::default_map(),
    };

    let state = Arc::new(AppState {
        pool,
        voices,
        api_key: args.api_key,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route(
            "/v1/text-to-speech/{voice_id}/stream-input",
            get(ws_handler),
        )
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    log::info!("Listening on ws://{}/v1/text-to-speech/{{voice_id}}/stream-input", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize, Debug)]
struct WsQuery {
    #[serde(default)]
    output_format: Option<String>,
    #[serde(rename = "xi-api-key", default)]
    xi_api_key: Option<String>,
    #[serde(default)]
    speaker_id: Option<u32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_audio_len_ms: Option<f32>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(voice_id): Path<String>,
    Query(q): Query<WsQuery>,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Optional API key auth: accept either the query param or `xi-api-key` header.
    if let Some(expected) = &state.api_key {
        let provided = q
            .xi_api_key
            .as_deref()
            .or_else(|| headers.get("xi-api-key").and_then(|v| v.to_str().ok()));
        if provided != Some(expected.as_str()) {
            return (StatusCode::UNAUTHORIZED, "invalid api key").into_response();
        }
    }

    let format = q
        .output_format
        .as_deref()
        .and_then(OutputFormat::parse)
        .unwrap_or(OutputFormat::Pcm24000);

    let Some(speaker_id) = state.voices.resolve(&voice_id) else {
        return (StatusCode::NOT_FOUND, format!("unknown voice_id '{voice_id}'")).into_response();
    };

    let mut params = GenParams::defaults(speaker_id);
    if let Some(s) = q.speaker_id {
        params.speaker_id = s;
    }
    if let Some(t) = q.temperature {
        params.temperature = t;
    }
    if let Some(k) = q.top_k {
        params.top_k = k;
    }
    if let Some(m) = q.max_audio_len_ms {
        params.max_audio_len_ms = m;
    }

    let state = state.clone();
    ws.on_upgrade(move |socket| async move {
        let guard = state.pool.checkout().await;
        let session = match Session::new(socket, guard, format, params) {
            Ok(s) => s,
            Err(e) => {
                log::error!("session init failed: {e}");
                return;
            }
        };
        if let Err(e) = session.run().await {
            log::warn!("session ended with error: {e}");
        }
    })
}

