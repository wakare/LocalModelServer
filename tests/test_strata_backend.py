from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_model_service.catalog import model_catalog, select_model
from local_model_service.config import ModelSettings, ServiceSettings
from local_model_service.runtime import ManagedRuntime, RuntimeFiles


def make_strata(tmp_path: Path, *, model_name="qwen3.8-flash-next-iq3_s") -> Path:
    root = tmp_path / "Strata"
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (root / ".venv" / "bin").mkdir()
    (root / ".venv" / "bin" / "python").write_bytes(b"python")
    engine = root / "engine" / "strata.exe"
    engine.parent.mkdir()
    engine.write_bytes(b"engine")
    main = tmp_path / "Strata-data" / "IQ3_S-00001-of-00002.gguf"
    main.parent.mkdir()
    main.write_bytes(b"weights")
    config = root / "strata-iq3_s.json"
    config.write_text(
        json.dumps(
            {
                "exe": str(engine),
                "args": ["--serve", "--native", str(main), "--max-context", "131072"],
                "model_name": model_name,
            }
        ),
        encoding="utf-8",
    )
    return config


def strata_settings(tmp_path: Path, config: Path) -> ServiceSettings:
    return ServiceSettings(
        assets_dir=str(tmp_path),
        model=ModelSettings(
            backend="strata",
            strata_config=str(config),
            model="qwen3.8-flash-next-iq3_s",
            model_preset="strata",
            server_speculative_type="none",
        ),
    )


def test_strata_command_runs_its_server_without_a_key_on_the_command_line(tmp_path) -> None:
    config = make_strata(tmp_path)
    settings = strata_settings(tmp_path, config).model
    settings.endpoint = "http://127.0.0.1:43210/v1/chat/completions"
    files = RuntimeFiles(settings, tmp_path)
    executable = files.resolve_executable()
    command = files.build_command(executable, files.resolve_model_path())
    assert command[1] == str(config.parent / "serve" / "server.py")
    assert command[2:6] == ["--engine", "strata", "--config", str(config)]
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "43210"]
    assert not any("key" in part.lower() for part in command)
    assert files.resolve_draft_model_path() is None


def test_strata_identity_follows_weights_engine_and_run_config(tmp_path) -> None:
    config = make_strata(tmp_path)
    runtime = ManagedRuntime(strata_settings(tmp_path, config), tmp_path / "service")
    assert runtime.model_path.name == "IQ3_S-00001-of-00002.gguf"
    assert runtime.draft_path is None and not runtime.fallback
    identities = {runtime.identity}
    runtime.model_path.write_bytes(b"other weights")
    runtime._refresh_identity()
    identities.add(runtime.identity)
    raw = json.loads(config.read_text(encoding="utf-8"))
    Path(raw["exe"]).write_bytes(b"other engine")
    runtime._refresh_identity()
    identities.add(runtime.identity)
    raw["args"].extend(("--kv", "int8"))
    config.write_text(json.dumps(raw), encoding="utf-8")
    runtime._refresh_identity()
    identities.add(runtime.identity)
    assert len(identities) == 4


def test_strata_backend_requires_a_run_config() -> None:
    settings = ServiceSettings(assets_dir="C:\\models", model=ModelSettings(backend="strata"))
    with pytest.raises(ValueError, match="strata_config"):
        settings.validate()
    settings.model.backend = "vllm"
    with pytest.raises(ValueError, match="backend"):
        settings.validate()


def test_catalog_switches_between_strata_and_llama_presets(tmp_path) -> None:
    config = make_strata(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "Qwen3.8-27B-Q4_K_M.gguf").write_bytes(b"qwen")
    runtime = tmp_path / "runtimes" / "llama" / "llama-server.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"llama")
    llama = ServiceSettings(
        assets_dir=str(tmp_path),
        model=ModelSettings(
            model_preset="qwen38_27b_q4_k_m",
            model="Qwen3.8-27B-Q4_K_M",
            model_path=str(models / "Qwen3.8-27B-Q4_K_M.gguf"),
            server_executable=str(runtime),
            server_speculative_type="none",
            strata_config=str(config),
        ),
    )
    catalog = model_catalog(llama)
    assert catalog["current"] == "qwen38_27b_q4_k_m"
    entry = next(item for item in catalog["models"] if item["id"] == "strata")
    assert entry["available"] and entry["model"] == "qwen3.8-flash-next-iq3_s"

    selected = select_model(llama, "strata")
    assert selected.model.backend == "strata"
    assert selected.model.model == "qwen3.8-flash-next-iq3_s"
    assert model_catalog(selected)["current"] == "strata"

    back = select_model(selected, "qwen38_27b_q4_k_m")
    assert back.model.backend == "llama"
    assert back.model.strata_config == str(config)
    assert back.model.model == "Qwen3.8-27B-Q4_K_M"


def test_catalog_hides_strata_when_its_run_config_is_missing(tmp_path) -> None:
    settings = ServiceSettings(
        assets_dir=str(tmp_path),
        model=ModelSettings(strata_config=str(tmp_path / "Strata" / "strata-x.json")),
    )
    (tmp_path / "models").mkdir()
    assert all(entry["id"] != "strata" for entry in model_catalog(settings)["models"])
