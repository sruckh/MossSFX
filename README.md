# MOSS-SoundEffect RunPod Serverless

> Scalable text-to-sound-effect inference server powered by [OpenMOSS-Team/MOSS-SoundEffect](https://huggingface.co/OpenMOSS/MOSS-SoundEffect) on RunPod Serverless GPU infrastructure.

A production-ready serverless worker that generates high-fidelity environmental sounds and sound effects from text descriptions, with automatic S3 upload and efficient resource utilization through lazy model loading.

## Features

- **Text-to-Sound Generation** — Generate ambient soundscapes and concrete sound effects from natural language descriptions
- **Duration Control** — Specify target duration in seconds; internally converted to RVQ audio tokens (1s ≈ 12.5 tokens)
- **WAV Output** — Native 24kHz WAV output, uploaded to S3-compatible storage with a presigned URL
- **S3 Integration** — Automatic upload to any S3-compatible backend (AWS, Backblaze B2, MinIO, etc.)
- **Lazy Model Loading** — Model loads on first request, minimizing cold-start overhead
- **Flash Attention 2** — Auto-detected on Ampere+ GPUs for faster inference and lower memory usage
- **24GB VRAM Guardrails** — Automatic token capping and one-shot OOM retry on constrained GPUs
- **Audio Tokenizer CPU Fallback** — Offloads audio tokenizer to CPU on OOM during preprocessing
- **RunPod Model Cache Awareness** — Checks RunPod's HuggingFace snapshot cache before downloading
- **Health Check Endpoint** — Inspect GPU availability, S3 config, and model presence
- **Network Volume Persistence** — All dependencies and model weights cached on a RunPod network volume

## Architecture

The worker consists of four files:

| File | Role |
|------|------|
| `handler.py` | RunPod serverless entry point — request validation, S3 upload, health check |
| `serverless_engine.py` | Inference engine — lazy model loading, generation, OOM handling |
| `config.py` | Environment-based configuration with validation |
| `bootstrap.sh` | First-boot setup — clones MOSS-TTS source, installs deps, optionally pre-downloads model |

![Architecture Diagram](docs/architecture.png)

### Data Flow

1. Client sends request with `text` and optional `duration_seconds`
2. Handler validates parameters and converts `duration_seconds` → audio tokens
3. Model loads lazily on first request (cached for subsequent requests)
4. Engine builds a `build_user_message(ambient_sound=text, tokens=N)` conversation
5. Model generates RVQ audio tokens via autoregressive decoding
6. Tokens are decoded to a waveform by the audio tokenizer/vocoder
7. Audio is saved as WAV and uploaded to S3
8. Response returned with presigned URL and metadata

### Model Source Resolution

On each load the engine tries sources in priority order:

1. **RunPod HF cache** — `/runpod-volume/huggingface-cache/hub` (standard HF snapshot format, populated by RunPod's model caching feature)
2. **Local volume** — `/runpod-volume/moss-sfx/models/OpenMOSS-Team/MOSS-SoundEffect`
3. **HuggingFace Hub** — downloaded on demand (requires `HF_TOKEN` for gated repos)

## Quick Start

### Prerequisites

- RunPod account with Serverless enabled
- S3-compatible storage bucket
- GPU with 24GB+ VRAM (48GB+ recommended for longer generations)

### Building the Container

```bash
git clone https://github.com/your-org/MossSFX.git
cd MossSFX

docker build -t your-registry/moss-sfx:latest .
docker push your-registry/moss-sfx:latest
```

### Deploying to RunPod

1. Create a new **Serverless Endpoint** in RunPod
2. Set the container image to your built image
3. Attach a **Network Volume** (recommended: 100GB+ for model weights and venv)
4. Set `BOOTSTRAP_DOWNLOAD_MODEL=true` and configure environment variables (see below)

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_ENDPOINT_URL` | S3-compatible API endpoint | `https://s3.amazonaws.com` |
| `S3_ACCESS_KEY_ID` | S3 access key | `AKIAIOSFODNN7EXAMPLE` |
| `S3_SECRET_ACCESS_KEY` | S3 secret key | `wJalrXUtnFEMI/...` |
| `S3_BUCKET_NAME` | Target bucket | `my-sfx-output` |
| `S3_REGION` | Signing region (for Backblaze B2 use your B2 region, e.g. `us-west-001`) | `us-east-1` |
| `S3_SIGNATURE_VERSION` | Signature algorithm | `s3v4` |
| `S3_ADDRESSING_STYLE` | S3 URL style (`path`, `virtual`, `auto`) | `path` |

### Recommended

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token (required for gated model access) | — |
| `BOOTSTRAP_DOWNLOAD_MODEL` | Pre-download model weights on first boot | `false` |
| `BOOTSTRAP_DOWNLOAD_AUDIO_TOKENIZER` | Pre-download audio tokenizer on first boot | `true` |
| `MODEL_REPO` | HuggingFace model repository | `OpenMOSS-Team/MOSS-SoundEffect` |
| `MODEL_REVISION` | Pinned commit SHA or branch | — |
| `AUDIO_TOKENIZER_REPO` | HuggingFace audio tokenizer repository | `OpenMOSS-Team/MOSS-Audio-Tokenizer` |
| `RUNPOD_HF_CACHE_DIR` | RunPod model cache mount path (read-only) | `/runpod-volume/huggingface-cache/hub` |
| `MOSS_DIR` | Root working directory on the network volume | `/runpod-volume/moss-sfx` |
| `RUNPOD_INIT_TIMEOUT` | Worker init timeout in seconds | `2400` |

### Decoding Defaults

These match the upstream recommended hyperparameters for MOSS-SoundEffect.

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_AUDIO_TEMPERATURE` | Sampling temperature (0–5) | `1.5` |
| `DEFAULT_AUDIO_TOP_P` | Nucleus sampling threshold (0–1) | `0.6` |
| `DEFAULT_AUDIO_TOP_K` | Top-k sampling (1–200) | `50` |
| `DEFAULT_AUDIO_REPETITION_PENALTY` | Repetition penalty (0.8–2.0) | `1.2` |
| `DEFAULT_MAX_NEW_TOKENS` | Max tokens to generate (128–8192) | `1024` |
| `MAX_DURATION_SECONDS` | Hard cap on `duration_seconds` input (`0` = no limit) | `0` |
| `OOM_TOKEN_CAP_24GB` | Auto-cap `max_new_tokens` on ~24GB GPUs | `1024` |
| `OOM_RETRY_MAX_NEW_TOKENS` | Token limit for one-shot OOM retry | `512` |
| `DEFAULT_DTYPE` | Torch dtype (`auto`, `bfloat16`, `float16`, `float32`) | `auto` |
| `DEFAULT_ATTN_IMPLEMENTATION` | Attention backend (`auto`, `flash_attention_2`, `sdpa`, `eager`) | `auto` |
| `DEFAULT_AUDIO_TOKENIZER_DEVICE` | Audio tokenizer device (`auto`, `cuda`, `cpu`) | `cuda` |
| `DEVICE` | Main inference device | `cuda` |

## Usage

### Basic Generation

```json
{
  "input": {
    "text": "Rain falling on a tin roof with distant thunder rumbling"
  }
}
```

### With Duration Control

```json
{
  "input": {
    "text": "Busy city street with traffic, honking horns, and pedestrians",
    "duration_seconds": 10
  }
}
```

### With Custom Decoding Parameters

```json
{
  "input": {
    "text": "Birds chirping in a quiet forest at dawn",
    "duration_seconds": 15,
    "audio_temperature": 1.5,
    "audio_top_p": 0.6,
    "audio_top_k": 50,
    "audio_repetition_penalty": 1.2
  }
}
```

### Health Check

```json
{
  "input": {
    "action": "health_check"
  }
}
```

## API Reference

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Description of the sound effect to generate (max 2,000 characters) |
| `duration_seconds` | float | No | Target duration in seconds — converted to audio tokens internally (1s ≈ 12.5 tokens) |
| `max_new_tokens` | int | No | Hard cap on generated tokens (128–8192, default: 1024; may be auto-capped on 24GB GPUs) |
| `audio_temperature` | float | No | Sampling temperature (0–5, default: 1.5) |
| `audio_top_p` | float | No | Nucleus sampling threshold (0–1, default: 0.6) |
| `audio_top_k` | int | No | Top-k sampling (1–200, default: 50) |
| `audio_repetition_penalty` | float | No | Repetition penalty (0.8–2.0, default: 1.2) |
| `session_id` | string | No | Custom job identifier (UUID auto-generated if omitted) |

### Response

**Success:**
```json
{
  "status": "completed",
  "filename": "550e8400-e29b-41d4-a716-446655440000.wav",
  "url": "https://s3.../presigned-url",
  "s3_key": "550e8400-e29b-41d4-a716-446655440000.wav",
  "metadata": {
    "sample_rate": 24000,
    "format": "wav",
    "duration_seconds": 9.856,
    "generation_time_seconds": 4.21,
    "device": "cuda",
    "model_repo": "OpenMOSS-Team/MOSS-SoundEffect",
    "tokens_hint": 125
  }
}
```

**Error:**
```json
{
  "error": "duration_seconds exceeds maximum of 30s",
  "error_type": "ValueError"
}
```

**Health check:**
```json
{
  "status": "healthy",
  "timestamp": 1745510400.0,
  "checks": {
    "configuration": { "status": "pass", "details": "ok" },
    "hardware": { "status": "pass", "details": "CUDA available=true, gpu=NVIDIA A40, memory=1.2/48.0GB" },
    "s3": { "status": "pass", "details": "configured=true" },
    "model": { "status": "pass", "details": "model_dir=/runpod-volume/moss-sfx/models/..., present=true" }
  }
}
```

### Output Format

| Property | Value |
|----------|-------|
| Container | WAV (uncompressed PCM) |
| Sample rate | 24,000 Hz |
| Channels | Mono |
| Bit depth | 32-bit float (normalized to 16-bit on save) |
| Delivery | S3 presigned URL (1-hour expiry) |

## Project Structure

```
MossSFX/
├── Dockerfile              # Container definition
├── bootstrap.sh            # First-boot setup and runtime launcher
├── config.py               # Environment-based configuration
├── handler.py              # RunPod request handler and S3 upload
├── serverless_engine.py    # MOSS-SoundEffect inference wrapper
└── requirements.txt        # Runtime dependency reference
```

## First Boot Process

On initial container start, `bootstrap.sh`:

1. Creates `/runpod-volume/moss-sfx/` directory tree on the network volume
2. Clones `OpenMOSS/MOSS-TTS` source from GitHub (contains MOSS-SoundEffect model code)
3. Creates a Python 3.12 virtual environment on the network volume
4. Installs PyTorch 2.9.1+cu128, flash-attn 2.8.3, and all inference dependencies via `uv`
5. Copies `handler.py`, `config.py`, `serverless_engine.py` from the Docker image into the source directory
6. Optionally pre-downloads MOSS-SoundEffect and audio tokenizer weights from HuggingFace
7. Starts the RunPod serverless handler

Subsequent boots skip steps 2–4 (sentinel file guards against re-installation) and proceed directly to step 5, then launch the handler within seconds.

## Performance Notes

- **First boot:** 10–20 minutes (dependency install + optional model download)
- **Warm start:** ~5 seconds (model already loaded in worker process)
- **Cold start:** 30–90 seconds (model load from network volume, no reinstall)
- **Generation time:** ~0.5–2× real-time depending on GPU and requested duration
- **GPU memory:** 24GB works with default token limits; 48GB+ gives comfortable headroom

## Troubleshooting

**OOM during generation**
- Reduce `max_new_tokens` or `duration_seconds`
- The worker auto-caps tokens on 24GB GPUs and retries once at a lower limit
- Use a 48GB+ GPU class for longer or more complex sounds

**Slow first request after cold start**
- Set `BOOTSTRAP_DOWNLOAD_MODEL=true` to pre-download the 8B model during bootstrap rather than on the first request

**Model not found / HuggingFace download failure**
- Verify `HF_TOKEN` is set and has access to `OpenMOSS-Team/MOSS-SoundEffect`
- Check `bootstrap.log` at `/runpod-volume/moss-sfx/bootstrap.log`
- Ensure the network volume has sufficient space (~20GB for model + tokenizer)

**Bootstrap log location**
```
/runpod-volume/moss-sfx/bootstrap.log
```

## Model

MOSS-SoundEffect is part of the [MOSS-TTS Family](https://github.com/OpenMOSS/MOSS-TTS) from [OpenMOSS](https://www.open-moss.com/). It uses the MossTTSDelay autoregressive architecture to generate RVQ audio tokens from text, then decodes them to a waveform via the MOSS Audio Tokenizer.

| Property | Value |
|----------|-------|
| Architecture | MossTTSDelay |
| Parameters | 8B |
| Precision | BF16 |
| Sample rate | 24,000 Hz |
| HuggingFace | [OpenMOSS-Team/MOSS-SoundEffect](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect) |

Please refer to the [model card](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_sound_effect_model_card.md) for usage terms and license details.

## Acknowledgments

- [OpenMOSS Team](https://github.com/OpenMOSS) for the MOSS-SoundEffect model
- [RunPod](https://www.runpod.io/) for serverless GPU infrastructure
