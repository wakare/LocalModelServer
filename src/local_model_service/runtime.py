from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlparse

from local_model_service.config import ModelSettings, ServiceSettings, is_loopback_endpoint
from local_model_service.presets import (
    FLASH_ATTENTION_MODES,
    SERVER_REASONING_MODES,
    SPECULATIVE_TYPES,
    LocalModelPreset,
    local_model_preset,
)
from local_model_service.win_job import ProcessJob


class LocalModelServerError(RuntimeError):
    pass


class RuntimeFiles:
    def __init__(self, settings: ModelSettings | None = None, data_dir: Path | None = None) -> None:
        # QObject's cooperative initializer also visits this mixin without arguments.
        if settings is not None:
            self.settings = settings
        if data_dir is not None:
            self.data_dir = data_dir

    def _strata(self) -> bool:
        # The reading assistant's settings reuse this mixin and have no backend field.
        return getattr(self.settings, "backend", "llama") == "strata"

    def resolve_executable(self) -> Path:
        if self._strata():
            python = strata_python(Path(self.settings.strata_config))
            if not python.is_file():
                raise LocalModelServerError(f"找不到 Strata 的 Python 环境：{python}")
            return python
        configured = self.settings.server_executable.strip()
        if configured:
            path = _expanded_path(configured)
            if path.is_file():
                return path
            discovered = shutil.which(configured)
            if discovered:
                return Path(discovered)
            raise LocalModelServerError(f"找不到 llama-server：{configured}")

        managed_runtimes = sorted(
            (self.data_dir / "runtimes").rglob("llama-server.exe"),
            key=lambda path: (
                int(
                    bool(self._draft_runtime_tag())
                    and self._draft_runtime_tag() in str(path).lower()
                ),
                *_runtime_executable_preference(path),
            ),
            reverse=True,
        )
        if managed_runtimes:
            return managed_runtimes[0]

        discovered = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if discovered:
            return Path(discovered)

        beside_app = Path(sys.executable).resolve().parent / "llama-server.exe"
        if beside_app.is_file():
            return beside_app

        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        packages = local_app_data / "Microsoft" / "WinGet" / "Packages"
        matches = sorted(packages.glob("ggml.llamacpp_*/*/llama-server.exe"))
        if not matches:
            matches = sorted(packages.glob("ggml.llamacpp_*/llama-server.exe"))
        if matches:
            return matches[-1]
        raise LocalModelServerError(
            "未找到 llama-server.exe；请先安装 llama.cpp，或在设置中填写程序路径"
        )

    def resolve_model_path(self) -> Path:
        if self._strata():
            path = strata_main_gguf(strata_run_config(Path(self.settings.strata_config)))
            if not path.is_file():
                raise LocalModelServerError(f"Strata 模型文件不存在：{path}")
            return path
        configured = self.settings.model_path.strip()
        if configured:
            path = _expanded_path(configured)
            if path.is_file():
                return path
            raise LocalModelServerError(f"本地模型文件不存在：{path}")

        model_dir = self.data_dir / "models"
        candidates = sorted(model_dir.glob("*.gguf"))
        if not candidates:
            raise LocalModelServerError(
                f"在 {model_dir} 中没有找到 GGUF 模型；可在设置中指定模型文件"
            )
        preset = local_model_preset(self.settings.model_preset)
        preset_matches_alias = bool(
            preset is not None
            and _normalized_model_name(self.settings.model)
            == _normalized_model_name(preset.model_alias)
        )
        if preset is not None and preset_matches_alias:
            preset_path = model_dir / preset.filename
            if preset_path.is_file():
                return preset_path
            raise LocalModelServerError(f"模型预设“{preset.display_name}”需要文件：{preset_path}")
        alias = _normalized_model_name(self.settings.model)
        matching = [
            path for path in candidates if alias and alias in _normalized_model_name(path.stem)
        ]
        if matching:
            candidates = matching
        elif len(candidates) > 1:
            raise LocalModelServerError(
                "模型目录中有多个 GGUF，但没有文件名匹配当前模型名称；请在设置中指定模型文件"
            )
        return min(candidates, key=_model_preference)

    def resolve_draft_model_path(self) -> Path | None:
        if self._strata():
            return None  # Strata runs its own MTP draft layer from its run config.
        speculative_type = self.settings.server_speculative_type.strip().lower()
        if speculative_type not in SPECULATIVE_TYPES:
            raise LocalModelServerError("推测解码类型必须是 " + "/".join(SPECULATIVE_TYPES))
        if speculative_type == "none":
            return None
        configured = self.settings.draft_model_path.strip()
        if configured:
            path = _expanded_path(configured)
            if path.is_file():
                return path
            raise LocalModelServerError(f"草稿模型文件不存在：{path}")
        preset = self._active_preset()
        if preset is None or not preset.draft_filename:
            raise LocalModelServerError("未配置草稿 GGUF 模型文件")
        path = self.data_dir / "models" / preset.draft_filename
        if not path.is_file():
            raise LocalModelServerError(f"模型预设“{preset.display_name}”需要草稿文件：{path}")
        return path

    def build_command(
        self,
        executable: Path,
        model_path: Path,
        *,
        draft_model_path: Path | None = None,
    ) -> list[str]:
        host, port = self._endpoint_address()
        if self._strata():
            config_path = Path(self.settings.strata_config)
            # The API key travels in STRATA_API_KEY so it never appears on a command line.
            return [
                str(executable),
                str(config_path.parent / "serve" / "server.py"),
                "--engine",
                "strata",
                "--config",
                str(config_path),
                "--host",
                host,
                "--port",
                str(port),
            ]
        reasoning = self.settings.server_reasoning.strip().lower()
        if reasoning not in SERVER_REASONING_MODES:
            raise LocalModelServerError(
                "llama-server reasoning 模式必须是 " + "/".join(SERVER_REASONING_MODES)
            )
        flash_attention = self.settings.server_flash_attention.strip().lower()
        if flash_attention not in FLASH_ATTENTION_MODES:
            raise LocalModelServerError(
                "Flash Attention 模式必须是 " + "/".join(FLASH_ATTENTION_MODES)
            )
        command = [
            str(executable),
            "-m",
            str(model_path),
            "--no-mmproj",
            "--alias",
            self.settings.model,
            "--host",
            host,
            "--port",
            str(port),
            "--cors-origins",
            "localhost",
            "--no-webui",
            "-ngl",
            str(self.settings.server_gpu_layers),
            "-c",
            str(self.settings.server_context_size),
            "-np",
            str(self.settings.server_parallel_slots),
            "--jinja",
            "--reasoning",
            reasoning,
            "-fa",
            flash_attention,
        ]
        if draft_model_path is not None:
            speculative_type = self.settings.server_speculative_type.strip().lower()
            if speculative_type not in SPECULATIVE_TYPES or speculative_type == "none":
                raise LocalModelServerError("草稿模型需要有效的推测解码类型")
            if self.settings.server_draft_max_tokens < 1:
                raise LocalModelServerError("草稿 token 数必须大于 0")
            command.extend(
                (
                    "--spec-type",
                    speculative_type,
                    "--spec-draft-model",
                    str(draft_model_path),
                    "--spec-draft-ngl",
                    str(self.settings.server_draft_gpu_layers),
                    "--spec-draft-n-max",
                    str(self.settings.server_draft_max_tokens),
                )
            )
        return command

    def _active_preset(self) -> LocalModelPreset | None:
        preset = local_model_preset(self.settings.model_preset)
        if preset is None:
            return None
        if _normalized_model_name(self.settings.model) != _normalized_model_name(
            preset.model_alias
        ):
            return None
        return preset

    def _draft_runtime_tag(self) -> str:
        preset = self._active_preset()
        if preset is None or self.settings.server_speculative_type == "none":
            return ""
        return preset.runtime_tag.strip().lower()

    def _endpoint_address(self) -> tuple[str, int]:
        if not is_loopback_endpoint(self.settings.endpoint):
            raise LocalModelServerError("本地模型 endpoint 必须是本机回环地址")
        parsed = urlparse(self.settings.endpoint)
        if parsed.scheme != "http" or not parsed.hostname:
            raise LocalModelServerError("托管的本地模型 endpoint 必须使用 http://")
        return parsed.hostname, parsed.port or 80


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class ManagedRuntime:
    """The broker worker is the sole owner of this lazily loaded backend."""

    def __init__(self, config: ServiceSettings, service_dir: Path) -> None:
        self.config = config
        self.service_dir = service_dir
        self.files = RuntimeFiles(config.model, Path(config.assets_dir))
        self.process: subprocess.Popen | None = None
        self.port = 0
        self.token = ""
        self._log = None
        self._job: ProcessJob | None = None
        self._lock = threading.RLock()
        self.state = "sleeping"
        self.reason = ""
        self.retry_at = 0.0
        self.restarts = 0
        self.fallback = False
        self._descriptor: dict[str, object] = {}
        self.identity = ""
        self.executable = self.files.resolve_executable().resolve()
        self.model_path = self.files.resolve_model_path().resolve()
        self.draft_path: Path | None = None
        if config.model.server_speculative_type != "none" and config.model.backend != "strata":
            try:
                self.draft_path = self.files.resolve_draft_model_path()
                tag = self.files._draft_runtime_tag()
                if (
                    tag
                    and not config.model.server_executable
                    and tag not in str(self.executable).lower()
                ):
                    raise LocalModelServerError(f"需要 {tag} 兼容运行时")
            except LocalModelServerError as error:
                if not config.model.server_draft_fallback:
                    raise
                self.fallback = True
                self.reason = str(error)
        self._refresh_identity()

    @property
    def effective_alias(self) -> str:
        alias = self.config.model.model
        if self.fallback:
            return (
                alias.removesuffix("-DFlash2") if alias.endswith("-DFlash2") else f"{alias}-plain"
            )
        return alias

    def _refresh_identity(self) -> None:
        parameters = asdict(self.config.model)
        for name in (
            "endpoint",
            "startup_timeout_seconds",
            "server_draft_fallback",
            "model_preset",
        ):
            parameters.pop(name)
        parameters["model"] = self.effective_alias
        draft = None if self.fallback else self.draft_path
        if draft is None:
            parameters.update(
                server_speculative_type="none",
                draft_model_path="",
                server_draft_max_tokens=0,
                server_draft_gpu_layers=0,
            )
        descriptor = {
            "version": 1,
            "effective_model": self.effective_alias,
            "main_file": _file_identity(self.model_path),
            "draft_file": _file_identity(draft) if draft else None,
            "runtime": _file_identity(self.executable),
            "runtime_libraries": [
                _file_identity(p) for p in sorted(self.executable.parent.glob("*.dll"))
            ],
            "parameters": parameters,
        }
        if self.config.model.backend == "strata":
            config_path = Path(self.config.model.strata_config)
            engine = _expanded_path(str(strata_run_config(config_path).get("exe", "")))
            descriptor["strata"] = {
                "config": _file_identity(config_path),
                "engine": _file_identity(engine) if engine.is_file() else None,
            }
        fingerprint = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self._lock:
            self._descriptor = descriptor
            self.identity = fingerprint

    def describe(self) -> dict[str, object]:
        with self._lock:
            if self.process is None and self.state in {"sleeping", "failed"}:
                self._refresh_identity()
            state = self.state
            if self.process is not None and self.process.poll() is not None and state == "ready":
                state = "failed"
            return {
                "identity": self.identity,
                "effective_model": self.effective_alias,
                "requested_model": self.config.model.model,
                "state": state,
                "fallback": self.fallback,
                "reason": self.reason,
                "pid": self.process.pid if self.process and self.process.poll() is None else None,
                "restarts": self.restarts,
                "retry_after_seconds": max(0, round(self.retry_at - time.monotonic(), 2)),
                "descriptor": self._descriptor,
            }

    def _backend_json(self, path: str, timeout: float = 1) -> dict:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request("GET", path, headers={"Authorization": f"Bearer {self.token}"})
            response = connection.getresponse()
            if response.status != 200:
                raise LocalModelServerError(f"backend {path}: HTTP {response.status}")
            return json.loads(response.read(1024 * 1024))
        finally:
            connection.close()

    def _verify_loaded_model(self) -> None:
        models = self._backend_json("/v1/models").get("data", [])
        if self.config.model.backend == "strata":
            if self.effective_alias not in {model.get("id") for model in models}:
                raise LocalModelServerError("backend 实际加载模型与服务身份不一致")
            return
        if len(models) != 1 or models[0].get("id") != self.effective_alias:
            raise LocalModelServerError("backend 实际加载模型与服务身份不一致")

    def ensure_ready(self, cancel: threading.Event, deadline: float) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self._backend_json("/health")
                self._verify_loaded_model()
                return
            except (OSError, ValueError, http.client.HTTPException, LocalModelServerError):
                self.stop()
        if time.monotonic() < self.retry_at:
            raise LocalModelServerError("模型恢复处于退避期，请稍后重试")
        self.stop()
        error: Exception = LocalModelServerError("模型未能就绪")
        attempts = 0
        while attempts < self.config.restart_attempts:
            try:
                self._refresh_identity()
                self._launch(cancel, deadline)
                return
            except (
                OSError,
                ValueError,
                http.client.HTTPException,
                LocalModelServerError,
            ) as caught:
                error = caught
                self.stop()
                if cancel.is_set() or time.monotonic() >= deadline:
                    break
                if (
                    self.draft_path
                    and not self.fallback
                    and self.config.model.server_draft_fallback
                ):
                    self.fallback = True
                    self.reason = f"DFlash2 失败，回退普通模型：{caught}"
                    logging.warning(self.reason)
                    self._refresh_identity()
                    continue
                attempts += 1
                self.restarts += 1
                if attempts < self.config.restart_attempts:
                    delay = self.config.recovery_backoff_seconds * 2 ** (attempts - 1)
                    cancel.wait(min(delay, max(0, deadline - time.monotonic())))
        with self._lock:
            self.state = "failed"
            self.reason = str(error)
            self.retry_at = time.monotonic() + self.config.recovery_backoff_seconds * 2**attempts
        raise LocalModelServerError(str(error)) from error

    def _launch(self, cancel: threading.Event, deadline: float) -> None:
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            self.port = port_socket.getsockname()[1]
        active = replace(
            self.config.model,
            model=self.effective_alias,
            endpoint=f"http://127.0.0.1:{self.port}/v1/chat/completions",
        )
        command = RuntimeFiles(active, Path(self.config.assets_dir)).build_command(
            self.executable,
            self.model_path,
            draft_model_path=None if self.fallback else self.draft_path,
        )
        self.service_dir.mkdir(parents=True, exist_ok=True)
        key_path = self.service_dir / "backend.key"
        self.token = secrets.token_urlsafe(32)
        key_path.write_text(self.token, encoding="utf-8")
        if os.name != "nt":
            key_path.chmod(0o600)
        strata = self.config.model.backend == "strata"
        environment = None
        working_dir = None
        if strata:
            environment = {**os.environ, "STRATA_API_KEY": self.token}
            working_dir = str(Path(self.config.model.strata_config).parent)
        else:
            command.extend(("--api-key-file", str(key_path), "--poll", "0", "--poll-batch", "0"))
        log_path = self.service_dir / "backend.log"
        if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
            for index in (2, 1):
                previous = log_path.with_suffix(f".log.{index}")
                if previous.exists():
                    previous.replace(log_path.with_suffix(f".log.{index + 1}"))
            log_path.replace(log_path.with_suffix(".log.1"))
        self._log = log_path.open("ab")
        self._job = ProcessJob()
        with self._lock:
            self.state = "loading"
        logging.info("Loading %s (identity %s)", self.effective_alias, self.identity)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=working_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._job.assign(self.process)
        load_deadline = min(deadline, time.monotonic() + active.startup_timeout_seconds)
        while not cancel.is_set() and time.monotonic() < load_deadline:
            if self.process.poll() is not None:
                raise LocalModelServerError(f"backend 启动退出：{self.process.returncode}")
            try:
                health = self._backend_json("/health", 0.5)
            except (OSError, ValueError, http.client.HTTPException, LocalModelServerError):
                cancel.wait(0.25)
                continue
            if strata and not health.get("loaded"):
                # Strata answers /health while its engine is still loading the experts.
                cancel.wait(0.5)
                continue
            self._verify_loaded_model()
            with self._lock:
                self.state = "ready"
                self.retry_at = 0
            return
        raise LocalModelServerError("模型加载已取消或超时")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._job:
            self._job.close()
            self._job = None
        if self._log:
            self._log.close()
            self._log = None
        with self._lock:
            self.process = None
            self.state = "sleeping"

    def mark_failed(self, reason: str) -> None:
        self.stop()
        with self._lock:
            self.state = "failed"
            self.reason = reason
            self.retry_at = max(
                self.retry_at, time.monotonic() + self.config.recovery_backoff_seconds
            )


def strata_run_config(path: Path) -> dict:
    """Strata setup's strata-<model>.json: the engine exe, its args and the served model name."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LocalModelServerError(f"无法读取 Strata 配置：{path}") from error
    if not isinstance(raw, dict) or not raw.get("exe") or not isinstance(raw.get("args"), list):
        raise LocalModelServerError(f"不是 Strata 模型配置：{path}")
    return raw


def strata_python(config_path: Path) -> Path:
    # Setup writes the config next to its private .venv in the Strata folder.
    root = config_path.parent / ".venv"
    return root / "Scripts" / "python.exe" if os.name == "nt" else root / "bin" / "python"


def strata_model_name(config: dict) -> str:
    return str(config.get("model_name") or "qwen3.8-flash-next")


def strata_main_gguf(config: dict) -> Path:
    args = [str(arg) for arg in config["args"]]
    for flag in ("--native", "--model", "-m"):
        if flag in args[:-1]:
            return _expanded_path(args[args.index(flag) + 1])
    raise LocalModelServerError("Strata 配置中没有主模型 GGUF")


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _normalized_model_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _runtime_executable_preference(path: Path) -> tuple[int, int, str]:
    """Prefer a CUDA-capable managed runtime, then its highest embedded build number."""
    has_cuda = int((path.parent / "ggml-cuda.dll").is_file())
    build_numbers = [int(value) for value in re.findall(r"b(\d+)", str(path).lower())]
    return has_cuda, max(build_numbers, default=0), str(path).lower()


def _model_preference(path: Path) -> tuple[int, str]:
    normalized = _normalized_model_name(path.stem)
    quantization_order = (
        "q3kxl",
        "iq4xs",
        "q4ks",
        "q4km",
        "q4",
    )
    rank = next(
        (index for index, marker in enumerate(quantization_order) if marker in normalized),
        len(quantization_order),
    )
    return rank, path.name.lower()
