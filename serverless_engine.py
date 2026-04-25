# coding=utf-8
# MOSS-SoundEffect RunPod Serverless Inference Engine

import gc
import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

import config

from transformers import AutoModel, AutoProcessor

log = logging.getLogger(__name__)

# Disable broken cuDNN SDPA backend, matching upstream usage.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

_MODEL_INIT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def _is_local_only_miss(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "couldn't connect to 'https://huggingface.co'" in text
        or "cannot find the requested files in the disk cache" in text
        or "local_files_only" in text
        or "localentrynotfounderror" in text
    )


class MossSFXInference:
    """MOSS-SoundEffect inference wrapper for RunPod serverless."""

    def __init__(self):
        self.model_repo = config.config.MODEL_REPO
        self.model_dir = config.config.MODEL_DIR
        self.device = config.config.device
        self.dtype = (config.config.default_dtype or "auto").lower()
        self.attn_implementation = config.config.default_attn_implementation
        self.audio_tokenizer_device_mode = config.config.default_audio_tokenizer_device
        self.model_revision = config.config.MODEL_REVISION
        self.audio_tokenizer_repo = config.config.AUDIO_TOKENIZER_REPO
        self.audio_tokenizer_dir = config.config.AUDIO_TOKENIZER_DIR
        self.audio_tokenizer_revision = config.config.AUDIO_TOKENIZER_REVISION
        self.oom_token_cap_24gb = int(config.config.oom_token_cap_24gb)
        self.oom_retry_max_new_tokens = int(config.config.oom_retry_max_new_tokens)

        self._model = None
        self._processor = None
        self._torch_device = None
        self._torch_dtype = None
        self._sample_rate = config.DEFAULT_SAMPLE_RATE

    @staticmethod
    def _is_complete_model_dir(path: Path) -> bool:
        if not path.exists():
            return False
        return (path / "config.json").exists() and (
            (path / "model.safetensors").exists()
            or (path / "model.safetensors.index.json").exists()
        )

    def _find_runpod_cached_snapshot(self, repo_id: str, revision: Optional[str] = None) -> Optional[Path]:
        cache_root = config.config.RUNPOD_HF_CACHE_DIR
        snapshots_dir = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.exists():
            return None
        if revision:
            pinned = snapshots_dir / revision
            if self._is_complete_model_dir(pinned):
                return pinned
        candidates = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in candidates:
            if self._is_complete_model_dir(snapshot):
                return snapshot
        return None

    def _resolve_repo_source(self, repo_id: str, local_dir: Path, revision: Optional[str] = None) -> Tuple[str, bool, str]:
        runpod_snapshot = self._find_runpod_cached_snapshot(repo_id, revision)
        if runpod_snapshot is not None:
            return str(runpod_snapshot), True, "runpod-cache"
        if self._is_complete_model_dir(local_dir):
            return str(local_dir), True, "local-volume"
        return repo_id, False, "huggingface-hub"

    def _resolve_dtype(self) -> torch.dtype:
        if self.dtype in {"auto", ""}:
            return torch.bfloat16 if self._torch_device.type == "cuda" else torch.float32
        if self.dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    def _resolve_attn_implementation(self) -> Optional[str]:
        requested = (self.attn_implementation or "").strip().lower()
        if requested == "none":
            return None
        if requested not in {"", "auto"}:
            return self.attn_implementation
        if (
            self._torch_device.type == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and self._torch_dtype in {torch.float16, torch.bfloat16}
        ):
            major, _ = torch.cuda.get_device_capability(self._torch_device)
            if major >= 8:
                return "flash_attention_2"
        if self._torch_device.type == "cuda":
            return "sdpa"
        return "eager"

    def _resolve_audio_tokenizer_device(self) -> torch.device:
        mode = (self.audio_tokenizer_device_mode or "auto").strip().lower()
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            if self._torch_device is not None and self._torch_device.type == "cuda":
                return self._torch_device
            log.warning("DEFAULT_AUDIO_TOKENIZER_DEVICE=cuda requested but CUDA unavailable; using CPU.")
            return torch.device("cpu")
        if self._torch_device is not None:
            return self._torch_device
        return torch.device("cpu")

    def _get_audio_tokenizer_device(self) -> torch.device:
        if not hasattr(self._processor, "audio_tokenizer") or self._processor.audio_tokenizer is None:
            return torch.device("cpu")
        try:
            return next(self._processor.audio_tokenizer.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with _MODEL_INIT_LOCK:
            if self._model is not None and self._processor is not None:
                return

            cuda_available = torch.cuda.is_available()
            requested_device = (self.device or "cpu").lower()
            if requested_device == "cuda" and not cuda_available:
                log.warning("CUDA requested but unavailable. Falling back to CPU.")

            self._torch_device = torch.device(
                "cuda" if requested_device == "cuda" and cuda_available else "cpu"
            )
            self._torch_dtype = self._resolve_dtype()

            model_source, local_files_only, source_kind = self._resolve_repo_source(
                self.model_repo, self.model_dir, self.model_revision
            )
            audio_tokenizer_source, _, audio_source_kind = self._resolve_repo_source(
                self.audio_tokenizer_repo, self.audio_tokenizer_dir, self.audio_tokenizer_revision
            )

            resolved_attn = self._resolve_attn_implementation()
            log.info(
                "Loading MOSS-SoundEffect model from %s (source=%s, local_only=%s)",
                model_source, source_kind, local_files_only,
            )
            log.info(
                "Loading audio tokenizer from %s (source=%s)",
                audio_tokenizer_source, audio_source_kind,
            )
            log.info("Device=%s dtype=%s attn=%s", self._torch_device, self._torch_dtype, resolved_attn)

            processor_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "codec_path": audio_tokenizer_source,
            }
            try:
                self._processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
            except Exception as exc:
                if source_kind != "huggingface-hub" and _is_local_only_miss(exc):
                    log.warning("Processor load missed local files; retrying via HuggingFace hub.")
                    processor_kwargs["codec_path"] = self.audio_tokenizer_repo
                    self._processor = AutoProcessor.from_pretrained(self.model_repo, **processor_kwargs)
                else:
                    raise

            if hasattr(self._processor, "audio_tokenizer"):
                tok_device = self._resolve_audio_tokenizer_device()
                self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(tok_device)
                log.info("Audio tokenizer moved to %s", tok_device)

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "dtype": self._torch_dtype,
                "local_files_only": local_files_only,
            }
            if self.model_revision and not local_files_only:
                model_kwargs["revision"] = self.model_revision
            if resolved_attn:
                model_kwargs["attn_implementation"] = resolved_attn

            try:
                self._model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self._torch_device)
            except Exception as exc:
                if local_files_only and _is_local_only_miss(exc):
                    log.warning("Local-only model load missed files; retrying with online fallback.")
                    model_kwargs["local_files_only"] = False
                    if self.model_revision:
                        model_kwargs["revision"] = self.model_revision
                    self._model = AutoModel.from_pretrained(model_source, **model_kwargs).to(self._torch_device)
                else:
                    raise

            self._model.eval()
            self._sample_rate = int(
                getattr(self._processor.model_config, "sampling_rate", config.DEFAULT_SAMPLE_RATE)
            )
            log.info("Model loaded; sample_rate=%s", self._sample_rate)

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    def _cap_max_new_tokens(self, requested: int) -> int:
        capped = int(requested)
        if self._torch_device is None or self._torch_device.type != "cuda":
            return capped
        try:
            total_gb = torch.cuda.get_device_properties(self._torch_device).total_memory / (1024 ** 3)
        except Exception:
            return capped
        if total_gb <= 24.5 and capped > self.oom_token_cap_24gb:
            log.warning(
                "Capping max_new_tokens from %s to %s for %.1fGB GPU.",
                capped, self.oom_token_cap_24gb, total_gb,
            )
            return self.oom_token_cap_24gb
        return capped

    def _generate_single(
        self,
        text: str,
        tokens: Optional[int],
        max_new_tokens: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
    ) -> Tuple[torch.Tensor, int]:
        msg_kwargs: Dict[str, Any] = {"ambient_sound": text}
        if tokens is not None:
            msg_kwargs["tokens"] = int(tokens)

        conversation = [[self._processor.build_user_message(**msg_kwargs)]]

        audio_tokenizer_prev_device: Optional[torch.device] = None
        used_cpu_fallback = False

        def _process():
            nonlocal audio_tokenizer_prev_device, used_cpu_fallback
            try:
                return self._processor(conversation, mode="generation")
            except Exception as proc_exc:
                if (
                    self._is_cuda_oom(proc_exc)
                    and self._torch_device is not None
                    and self._torch_device.type == "cuda"
                    and hasattr(self._processor, "audio_tokenizer")
                ):
                    audio_tokenizer_prev_device = self._get_audio_tokenizer_device()
                    if audio_tokenizer_prev_device.type == "cuda":
                        log.warning("CUDA OOM during processor step; retrying with audio tokenizer on CPU.")
                        self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(torch.device("cpu"))
                        torch.cuda.empty_cache()
                        used_cpu_fallback = True
                        return self._processor(conversation, mode="generation")
                raise

        def _decode(outputs) -> torch.Tensor:
            messages = self._processor.decode(outputs)
            if not messages:
                raise RuntimeError("Model returned no messages")
            audio = messages[0].audio_codes_list[0]
            tensor = audio.detach().float().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio, dtype=torch.float32)
            return tensor.reshape(-1) if tensor.dim() > 1 else tensor

        capped_tokens = self._cap_max_new_tokens(max_new_tokens)
        generate_kwargs = {
            "max_new_tokens": capped_tokens,
            "audio_temperature": float(audio_temperature),
            "audio_top_p": float(audio_top_p),
            "audio_top_k": int(audio_top_k),
            "audio_repetition_penalty": float(audio_repetition_penalty),
        }

        try:
            with torch.inference_mode():
                batch = _process()
                input_ids = batch["input_ids"].to(self._torch_device)
                attention_mask = batch["attention_mask"].to(self._torch_device)
                outputs = self._model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
                )
                audio_tensor = _decode(outputs)
        except Exception as exc:
            if self._is_cuda_oom(exc) and self._torch_device is not None and self._torch_device.type == "cuda":
                retry_tokens = max(64, min(capped_tokens, self.oom_retry_max_new_tokens))
                if retry_tokens < capped_tokens:
                    log.warning(
                        "CUDA OOM at max_new_tokens=%s. Retrying once with max_new_tokens=%s.",
                        capped_tokens, retry_tokens,
                    )
                    torch.cuda.empty_cache()
                    with torch.inference_mode():
                        batch = _process()
                        input_ids = batch["input_ids"].to(self._torch_device)
                        attention_mask = batch["attention_mask"].to(self._torch_device)
                        generate_kwargs["max_new_tokens"] = retry_tokens
                        outputs = self._model.generate(
                            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
                        )
                        audio_tensor = _decode(outputs)
                else:
                    torch.cuda.empty_cache()
                    raise
            else:
                if self._torch_device is not None and self._torch_device.type == "cuda":
                    torch.cuda.empty_cache()
                raise
        finally:
            if (
                used_cpu_fallback
                and audio_tokenizer_prev_device is not None
                and audio_tokenizer_prev_device.type == "cuda"
                and hasattr(self._processor, "audio_tokenizer")
            ):
                try:
                    self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(audio_tokenizer_prev_device)
                    log.info("Restored audio tokenizer to %s after CPU fallback.", audio_tokenizer_prev_device)
                except Exception as restore_exc:
                    log.warning("Could not restore audio tokenizer to CUDA: %s", restore_exc)
            if self._torch_device is not None and self._torch_device.type == "cuda":
                torch.cuda.empty_cache()

        gc.collect()
        return audio_tensor, self._sample_rate

    def generate_audio(
        self,
        text: str,
        tokens: Optional[int] = None,
        max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS,
        audio_temperature: float = config.DEFAULT_AUDIO_TEMPERATURE,
        audio_top_p: float = config.DEFAULT_AUDIO_TOP_P,
        audio_top_k: int = config.DEFAULT_AUDIO_TOP_K,
        audio_repetition_penalty: float = config.DEFAULT_AUDIO_REPETITION_PENALTY,
    ) -> Tuple[torch.Tensor, int]:
        self._load_model()
        with _INFER_LOCK:
            return self._generate_single(
                text=text,
                tokens=tokens,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )


_engine_instance: Optional[MossSFXInference] = None
_engine_lock = threading.Lock()


def get_inference_engine() -> MossSFXInference:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = MossSFXInference()
    return _engine_instance
