from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from local_model_service.config import ModelSettings, ServiceSettings
from local_model_service.presets import LOCAL_MODEL_PRESETS, LocalModelPreset
from local_model_service.runtime import (
    LocalModelServerError,
    RuntimeFiles,
    strata_main_gguf,
    strata_model_name,
    strata_run_config,
)

STRATA_PRESET = "strata"


def _preset_settings(config: ServiceSettings, preset: LocalModelPreset) -> ModelSettings:
    model_dir = Path(config.assets_dir) / "models"
    executable = config.model.server_executable
    if preset.runtime_tag and preset.runtime_tag not in executable.lower():
        executable = ""  # Discover a compatible draft runtime before considering fallback.
    return replace(
        config.model,
        backend="llama",
        model_preset=preset.preset_id,
        model=preset.model_alias,
        model_path=str(model_dir / preset.filename),
        server_executable=executable,
        server_gpu_layers=preset.gpu_layers,
        server_context_size=preset.context_size,
        server_reasoning=preset.server_reasoning,
        server_flash_attention=preset.flash_attention,
        draft_model_path=str(model_dir / preset.draft_filename) if preset.draft_filename else "",
        server_draft_gpu_layers=preset.draft_gpu_layers,
        server_speculative_type=preset.speculative_type,
        server_draft_max_tokens=preset.draft_max_tokens,
    )


def _entry(selection_id: str, name: str, settings: ModelSettings, config: ServiceSettings) -> dict:
    main = Path(settings.model_path)
    missing = [] if main.is_file() else [main.name]
    warnings = []
    files = RuntimeFiles(settings, Path(config.assets_dir))
    try:
        executable = files.resolve_executable()
        tag = files._draft_runtime_tag()
        if tag and not settings.server_executable and tag not in str(executable).lower():
            warnings.append(f"缺少 {tag} 草稿运行时，将使用普通推理")
    except LocalModelServerError:
        missing.append("llama-server 运行时")
    if settings.draft_model_path and not Path(settings.draft_model_path).is_file():
        warnings.append(f"缺少草稿文件 {Path(settings.draft_model_path).name}，将使用普通推理")
    if warnings and not settings.server_draft_fallback:
        missing.extend(warnings)
        warnings = []
    return {
        "id": selection_id,
        "name": name,
        "model": settings.model,
        "filename": main.name,
        "size_bytes": main.stat().st_size if main.is_file() else None,
        "preset": settings.model_preset,
        "available": not missing,
        "missing": missing,
        "warning": "；".join(warnings),
    }


def _choices(config: ServiceSettings) -> list[tuple[dict, ModelSettings]]:
    choices = []
    known_files = {preset.filename.casefold() for preset in LOCAL_MODEL_PRESETS}
    draft_files = {preset.draft_filename.casefold() for preset in LOCAL_MODEL_PRESETS}
    for preset in LOCAL_MODEL_PRESETS:
        settings = _preset_settings(config, preset)
        choices.append((_entry(preset.preset_id, preset.display_name, settings, config), settings))
    # Additional local GGUFs are selectable without accepting arbitrary paths from HTTP.
    for path in sorted((Path(config.assets_dir) / "models").glob("*.gguf")):
        if path.name.casefold() in known_files | draft_files or "dflash" in path.name.casefold():
            continue
        settings = replace(
            config.model,
            backend="llama",
            model_preset="custom",
            model=path.stem,
            model_path=str(path.resolve()),
            server_gpu_layers=99,
            server_reasoning="off" if "qwen3.8" in path.name.casefold() else "auto",
            server_flash_attention="auto",
            draft_model_path="",
            server_speculative_type="none",
            server_draft_max_tokens=0,
        )
        choices.append((_entry("file:" + path.name, path.stem, settings, config), settings))
    strata = _strata_settings(config)
    if strata is not None:
        name = f"Strata · {strata.model}"
        choices.append((_entry(STRATA_PRESET, name, strata, config), strata))
    return choices


def _strata_settings(config: ServiceSettings) -> ModelSettings | None:
    """The Strata run config named in service.json stays selectable while another model runs."""
    config_path = config.model.strata_config
    if not config_path or not Path(config_path).is_file():
        return None
    try:
        run_config = strata_run_config(Path(config_path))
        main = strata_main_gguf(run_config)
    except LocalModelServerError:
        return None
    return replace(
        config.model,
        backend="strata",
        model_preset=STRATA_PRESET,
        model=strata_model_name(run_config),
        model_path=str(main),
        server_executable="",
        draft_model_path="",
        server_speculative_type="none",
        server_draft_max_tokens=0,
    )


def model_catalog(config: ServiceSettings) -> dict:
    """Read filenames and stat metadata only; never parse weights or start a backend."""
    choices = _choices(config)
    current = "current"
    files = RuntimeFiles(config.model, Path(config.assets_dir))
    try:
        current_path = files.resolve_model_path().resolve()
    except LocalModelServerError:
        current_path = Path(config.model.model_path)
    for entry, settings in choices:
        if (
            settings.model == config.model.model
            and Path(settings.model_path).resolve() == current_path
            and settings.model_preset == config.model.model_preset
        ):
            current = entry["id"]
            break
    entries = [entry for entry, _ in choices]
    if current == "current":
        settings = replace(config.model, model_path=str(current_path))
        entries.insert(0, _entry("current", config.model.model + "（当前配置）", settings, config))
    return {"current": current, "models": entries}


def select_model(config: ServiceSettings, selection_id: str) -> ServiceSettings:
    for entry, settings in _choices(config):
        if entry["id"] != selection_id:
            continue
        if not entry["available"]:
            raise ValueError("模型不可用，缺少：" + "、".join(entry["missing"]))
        result = replace(config, model=settings)
        result.validate()
        return result
    raise ValueError("请选择列表中已安装的模型")
