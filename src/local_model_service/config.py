from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from local_model_service.presets import DEFAULT_LOCAL_MODEL_PRESET

BACKENDS = ("llama", "strata")


def is_loopback_endpoint(endpoint: str) -> bool:
    try:
        parsed = urlparse(endpoint)
        return parsed.scheme in {"http", "https"} and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
    except ValueError:
        return False


def default_service_dir() -> Path:
    override = os.environ.get("LOCAL_MODEL_SERVICE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        for directory in (executable_dir, executable_dir.parent):
            if (directory / "service.json").is_file():
                return directory
    # User-profile storage avoids MSIX redirecting LOCALAPPDATA into its own cache.
    return Path.home() / ".local-model-service"


@dataclass(slots=True)
class ModelSettings:
    model_preset: str = DEFAULT_LOCAL_MODEL_PRESET
    model: str = "Qwen3.8-27B-Q4_K_M-DFlash2"
    model_path: str = ""
    server_executable: str = ""
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    server_gpu_layers: int = 99
    server_context_size: int = 4096
    server_parallel_slots: int = 1
    server_reasoning: str = "off"
    server_flash_attention: str = "on"
    draft_model_path: str = ""
    server_draft_gpu_layers: int = 99
    server_speculative_type: str = "draft-dflash"
    server_draft_max_tokens: int = 7
    server_draft_fallback: bool = True
    startup_timeout_seconds: float = 180
    # "strata" runs Strata's own server from its run config instead of llama-server.
    backend: str = "llama"
    strata_config: str = ""


@dataclass(slots=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8080
    assets_dir: str = ""
    idle_seconds: float = 60
    queue_capacity: int = 32
    request_timeout_seconds: float = 300
    inference_timeout_seconds: float = 90
    restart_attempts: int = 2
    recovery_backoff_seconds: float = 2
    max_body_bytes: int = 4 * 1024 * 1024
    api_key_env: str = ""
    model: ModelSettings = field(default_factory=ModelSettings)

    def validate(self) -> None:
        if self.host != "127.0.0.1":
            raise ValueError("服务只能绑定 127.0.0.1")
        if not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("服务端口无效")
        for name in (
            "idle_seconds",
            "request_timeout_seconds",
            "inference_timeout_seconds",
            "recovery_backoff_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.queue_capacity < 1 or self.queue_capacity > 256:
            raise ValueError("queue_capacity 必须为 1–256")
        if not 1 <= self.restart_attempts <= 5:
            raise ValueError("restart_attempts 必须为 1–5")
        if not 1 <= self.max_body_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_body_bytes 无效")
        if not self.model.model.strip() or any(c in self.model.model for c in "\r\n"):
            raise ValueError("模型别名无效")
        if self.model.server_parallel_slots != 1:
            raise ValueError("服务使用统一队列，backend parallel slots 必须为 1")
        if self.model.startup_timeout_seconds <= 0:
            raise ValueError("模型加载超时必须大于 0")
        if self.model.backend not in BACKENDS:
            raise ValueError("backend 必须是 " + "/".join(BACKENDS))
        if self.model.strata_config and not Path(self.model.strata_config).is_absolute():
            raise ValueError("strata_config 必须是绝对路径")
        if self.model.backend == "strata" and not self.model.strata_config:
            raise ValueError("Strata 后端需要 strata_config")
        if self.model.server_context_size < 512:
            raise ValueError("模型 context size 至少为 512")
        if not self.assets_dir or not Path(self.assets_dir).is_absolute():
            raise ValueError("assets_dir 必须是模型资源的绝对路径")

    @classmethod
    def load(cls, path: Path) -> ServiceSettings:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["model"] = ModelSettings(**raw.get("model", {}))
        result = cls(**raw)
        result.validate()
        return result

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
