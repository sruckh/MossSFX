# coding=utf-8
# MOSS-SoundEffect RunPod Serverless Handler

import argparse
import io
import time
import traceback
import wave
from typing import Any, Dict, Generator, Optional
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import runpod
import torch
from botocore.config import Config as BotoConfig

import config as config_module
from config import config, TOKENS_PER_SECOND
from serverless_engine import get_inference_engine

log = runpod.RunPodLogger()


def get_s3_client():
    missing = [
        name
        for name in ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME"]
        if not getattr(config, name)
    ]
    if missing:
        raise RuntimeError(f"Missing S3 configuration: {', '.join(missing)}")

    endpoint_host = urlparse(config.S3_ENDPOINT_URL).hostname or ""
    region = config.S3_REGION
    if endpoint_host.endswith(".backblazeb2.com"):
        parts = endpoint_host.split(".")
        if len(parts) >= 4 and parts[0] == "s3":
            inferred = parts[1]
            if region in {"", "us-east-1"}:
                region = inferred
                log.info(f"Inferred S3 region '{region}' from Backblaze endpoint.")

    client_cfg = BotoConfig(
        region_name=region,
        signature_version=config.S3_SIGNATURE_VERSION,
        s3={"addressing_style": config.S3_ADDRESSING_STYLE},
    )
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        config=client_cfg,
    )


def upload_to_s3(audio: torch.Tensor, sample_rate: int, session_id: str) -> Dict[str, str]:
    filename = f"{session_id}.wav"
    buf = io.BytesIO()
    _write_wav(buf, audio, sample_rate)
    buf.seek(0)

    s3 = get_s3_client()
    s3.upload_fileobj(buf, config.S3_BUCKET_NAME, filename, ExtraArgs={"ContentType": "audio/wav"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": config.S3_BUCKET_NAME, "Key": filename},
        ExpiresIn=3600,
    )
    return {"filename": filename, "url": url, "s3_key": filename}


def _write_wav(buf: io.BytesIO, audio: torch.Tensor, sample_rate: int) -> None:
    waveform = audio.detach()
    if waveform.ndim == 2:
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        else:
            waveform = waveform.transpose(0, 1)
    elif waveform.ndim != 1:
        raise ValueError(f"Audio tensor must be 1D or 2D, got shape {tuple(waveform.shape)}")

    samples = (
        waveform.to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(dtype=torch.int16)
        .contiguous()
        .numpy()
    )
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())


def handle_health_check() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    checks["configuration"] = (
        {"status": "fail", "details": str(config.validation_errors)}
        if config.validation_errors
        else {"status": "pass", "details": "ok"}
    )

    cuda_available = torch.cuda.is_available()
    hw_details = f"CUDA available={cuda_available}"
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)
        hw_details += f", gpu={props.name}, memory={used:.1f}/{total:.1f}GB"
    checks["hardware"] = {"status": "pass", "details": hw_details}

    s3_ok = all([config.S3_ENDPOINT_URL, config.S3_ACCESS_KEY_ID, config.S3_SECRET_ACCESS_KEY, config.S3_BUCKET_NAME])
    checks["s3"] = {"status": "pass" if s3_ok else "warn", "details": f"configured={s3_ok}"}

    model_present = (config.MODEL_DIR / "config.json").exists()
    checks["model"] = {
        "status": "pass" if model_present else "warn",
        "details": f"model_dir={config.MODEL_DIR}, present={model_present}",
    }

    overall = "healthy" if all(c["status"] in {"pass", "warn"} for c in checks.values()) else "degraded"
    return {"status": overall, "timestamp": time.time(), "checks": checks}


def _parse_float(value: Any, name: str, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def _parse_int(value: Any, name: str, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def handler(event: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    job_input = event.get("input", {})

    if job_input.get("action") == "health_check":
        yield handle_health_check()
        return

    if not config.validate():
        yield {"error": "Server misconfigured", "error_type": "ConfigurationError"}
        return

    # --- text ---
    text = job_input.get("text", "")
    if not isinstance(text, str) or not text.strip():
        yield {"error": "text is required and must be a non-empty string", "error_type": "ValueError"}
        return
    text = text.strip()
    if len(text) > 2000:
        yield {"error": f"text too long (max 2000 chars, got {len(text)})", "error_type": "ValueError"}
        return

    # --- duration_seconds → tokens ---
    tokens: Optional[int] = None
    raw_duration = job_input.get("duration_seconds")
    if raw_duration is not None:
        try:
            dur = float(raw_duration)
        except (TypeError, ValueError):
            yield {"error": "duration_seconds must be a number", "error_type": "ValueError"}
            return
        if dur <= 0:
            yield {"error": "duration_seconds must be greater than 0", "error_type": "ValueError"}
            return
        max_dur = config.max_duration_seconds
        if max_dur > 0 and dur > max_dur:
            yield {"error": f"duration_seconds exceeds maximum of {max_dur}s", "error_type": "ValueError"}
            return
        tokens = max(1, round(dur * TOKENS_PER_SECOND))

    # --- decoding parameters ---
    try:
        max_new_tokens = _parse_int(
            job_input.get("max_new_tokens", config.default_max_new_tokens),
            "max_new_tokens", 128, 8192,
        )
        audio_temperature = _parse_float(
            job_input.get("audio_temperature", config.default_audio_temperature),
            "audio_temperature", 0.0, 5.0,
        )
        audio_top_p = _parse_float(
            job_input.get("audio_top_p", config.default_audio_top_p),
            "audio_top_p", 0.0, 1.0,
        )
        audio_top_k = _parse_int(
            job_input.get("audio_top_k", config.default_audio_top_k),
            "audio_top_k", 1, 200,
        )
        audio_repetition_penalty = _parse_float(
            job_input.get("audio_repetition_penalty", config.default_audio_repetition_penalty),
            "audio_repetition_penalty", 0.8, 2.0,
        )
    except ValueError as exc:
        yield {"error": str(exc), "error_type": "ValueError"}
        return

    session_id = str(job_input.get("session_id") or uuid4())

    # --- inference ---
    engine = get_inference_engine()
    start = time.time()
    try:
        audio, sample_rate = engine.generate_audio(
            text=text,
            tokens=tokens,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )
    except Exception as exc:
        log.error(f"Inference failed [{session_id}]: {exc}\n{traceback.format_exc()}")
        yield {"error": str(exc), "error_type": type(exc).__name__}
        return

    elapsed = time.time() - start
    duration = audio.shape[-1] / sample_rate if sample_rate > 0 else 0.0

    # --- S3 upload ---
    try:
        s3_result = upload_to_s3(audio, sample_rate, session_id)
    except Exception as exc:
        log.error(f"S3 upload failed [{session_id}]: {exc}\n{traceback.format_exc()}")
        yield {"error": f"S3 upload failed: {exc}", "error_type": type(exc).__name__}
        return

    yield {
        "status": "completed",
        "filename": s3_result["filename"],
        "url": s3_result["url"],
        "s3_key": s3_result["s3_key"],
        "metadata": {
            "sample_rate": sample_rate,
            "format": "wav",
            "duration_seconds": round(duration, 3),
            "generation_time_seconds": round(elapsed, 3),
            "device": str(engine._torch_device),
            "model_repo": config.MODEL_REPO,
            "tokens_hint": tokens,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true", help="Load model and exit")
    args = parser.parse_args()

    if args.warmup:
        log.info("Warmup: loading model...")
        get_inference_engine()._load_model()
        log.info("Warmup complete.")
    else:
        runpod.serverless.start({"handler": handler})
