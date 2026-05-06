# Sesame CSM-1B WebSocket TTS

Real-time TTS service exposing **Sesame CSM-1B** over a WebSocket protocol
that mirrors the ElevenLabs `stream-input` API. Designed for phone-call
scenarios: clients stream text in, get audio chunks back as they're generated.

Built on [`cartesia-one/csm.rs`](https://github.com/cartesia-one/csm.rs) for
the underlying Candle-based inference (CUDA / cuDNN / Metal / Accelerate / MKL).

## Layout

```
csm-core/   vendored from cartesia-one/csm.rs (model + Mimi decoder)
csm-ws/     this project — axum WebSocket server
Dockerfile  CUDA + cuDNN runtime image for prod
```

## Build

Dev (macOS, Apple Silicon GPU):
```bash
cargo build --release -p csm-ws --features metal
```

Prod (Linux + NVIDIA):
```bash
cargo build --release -p csm-ws --features cudnn
# or just CUDA without cuDNN:
cargo build --release -p csm-ws --features cuda
```

## Run

```bash
./target/release/csm-ws \
    --model-id sesame/csm-1b \
    --port 8080 \
    --pool-size 1
```

Models are pulled from the Hugging Face Hub on first use and cached under
`~/.cache/huggingface/`. For a quantized GGUF model:

```bash
./target/release/csm-ws \
    --model-id cartesia/sesame-csm-1b-gguf \
    --model-file q8.gguf
```

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--host` | `0.0.0.0` | |
| `--port` | `8080` | |
| `--api-key` | unset | If set, clients must pass `xi-api-key` header or query. |
| `--pool-size` | `1` | Number of pre-loaded generator instances = max concurrent sessions. CSM-1B fp16 ≈ 2 GB VRAM each. |
| `--cpu` | false | Force CPU inference. |
| `--dtype` | auto | `f32` / `f16` / `bf16`. Defaults to f16 on CUDA, f32 elsewhere. |
| `--model-id` | none | HF Hub model id. |
| `--weights-path` | none | Local `.safetensors` or `.gguf`. Highest precedence. |
| `--voices-file` | none | JSON file mapping `voice_id` strings to numeric speaker ids. |

### Voice map file (`--voices-file`)

```json
{
  "rachel": 0,
  "adam": 1,
  "sales-bot-en-us": 2
}
```

Without this flag, the defaults `default`, `speaker_0`..`speaker_3` are
recognized, plus any numeric `voice_id` (`"0"`, `"1"`, ...) passes through.

## WebSocket protocol

### Connect

```
ws://host:8080/v1/text-to-speech/{voice_id}/stream-input?output_format={format}
```

Query params:
- `output_format` — one of `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100`, `ulaw_8000`. Default `pcm_24000` (CSM's native rate, no resampling).
- `xi-api-key` — required if server was started with `--api-key`.
- `speaker_id`, `temperature`, `top_k`, `max_audio_len_ms` — optional generation overrides.

### Client → server messages

```jsonc
// Append text. Server buffers until a sentence boundary, then generates.
{"text": "Hello there. "}

// Force the server to generate whatever's buffered now.
{"text": "more text ", "flush": true}

// Per-message overrides (Sesame extension, not in ElevenLabs API):
{"text": "...", "speaker_id": 2, "temperature": 0.6, "top_k": 50}

// End-of-stream — flushes any remaining text and closes.
{"text": ""}
```

### Server → client messages

```jsonc
// Audio chunk in the requested output_format, base64 encoded.
{"audio": "<base64>", "isFinal": false}

// Sent once after the final chunk for a generation.
{"audio": null, "isFinal": true}
```

### Phone-call setup (Twilio Media Streams)

Connect with `output_format=ulaw_8000`. The base64 `audio` field decodes
directly to the bytes Twilio expects in its `media.payload`.

## Concurrency model

CSM-1B is autoregressive and keeps a per-session KV cache. Each WebSocket
holds one generator instance for its lifetime. `--pool-size` is therefore
the **max number of concurrent calls** the server can handle.

For an L4 / A10G (24 GB) you can fit ~8 fp16 instances. For an L40 / H100,
substantially more. Set `--pool-size` to match your VRAM budget.

## Latency notes

- CSM-1B generates audio at ~12.5 frames/sec internally (80 ms per Mimi frame).
- The server buffers ~4 frames (≈320 ms) before decoding the first audio
  chunk. This is `--buffer-size` in csm-core terms; configurable per-session.
- First-byte latency on an RTX 4090: ~250 ms typical.
- On Metal (M-series Mac dev): faster than realtime for fp32 weights.

## Docker

```bash
docker build -t sesame-csm-ws --build-arg CUDA_COMPUTE_CAP=89 .
docker run --gpus all -p 8080:8080 \
    -v $HOME/.cache/huggingface:/home/csm/.cache/huggingface \
    sesame-csm-ws --model-id sesame/csm-1b
```

`CUDA_COMPUTE_CAP` values: `80` (A100), `86` (RTX 30xx), `89` (RTX 40xx / L4 / L40), `90` (H100).

## Quick client (Python)

```python
import asyncio, json, base64, websockets

async def main():
    uri = "ws://localhost:8080/v1/text-to-speech/default/stream-input?output_format=pcm_24000"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"text": "Hello from sesame. How are you today? "}))
        await ws.send(json.dumps({"text": ""}))  # EOS
        with open("out.pcm", "wb") as f:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("audio"):
                    f.write(base64.b64decode(msg["audio"]))
                if msg.get("isFinal"):
                    break

asyncio.run(main())
```

Play with: `ffplay -f s16le -ar 24000 -ac 1 out.pcm`

## License

AGPL-3.0, inherited from `csm.rs`.
