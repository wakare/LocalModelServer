from __future__ import annotations

from dataclasses import dataclass

CUSTOM_LOCAL_MODEL_PRESET = "custom"
DEFAULT_LOCAL_MODEL_PRESET = "qwen38_27b_q4_k_m_dflash2"
SERVER_REASONING_MODES = ("auto", "off", "on")
FLASH_ATTENTION_MODES = ("auto", "on", "off")
SPECULATIVE_TYPES = ("none", "draft-dflash")


@dataclass(frozen=True, slots=True)
class LocalModelPreset:
    preset_id: str
    display_name: str
    model_alias: str
    filename: str
    gpu_layers: int
    context_size: int
    grammar_input_variant: str
    server_reasoning: str
    description: str
    flash_attention: str = "auto"
    draft_filename: str = ""
    draft_gpu_layers: int = 99
    speculative_type: str = "none"
    draft_max_tokens: int = 0
    runtime_tag: str = ""


LOCAL_MODEL_PRESETS = (
    LocalModelPreset(
        preset_id="qwen35_a3b_q3_k_xl",
        display_name="Qwen3.5 35B-A3B · Q3_K_XL（历史语法对照）",
        model_alias="Qwen3.5-35B-A3B-Q3_K_XL",
        filename="Qwen_Qwen3.5-35B-A3B-Q3_K_XL.gguf",
        gpu_layers=36,
        context_size=4096,
        grammar_input_variant="baseline",
        server_reasoning="auto",
        description=(
            "从语法发现推荐候选降级为历史回归与量化对照；"
            "不据当前 reviewed 审计集评价基础模型整体能力。"
        ),
    ),
    LocalModelPreset(
        preset_id="qwen35_a3b_iq4_xs",
        display_name="Qwen3.5 35B-A3B · IQ4_XS（历史语法对照）",
        model_alias="Qwen3.5-35B-A3B-IQ4_XS",
        filename="Qwen_Qwen3.5-35B-A3B-IQ4_XS.gguf",
        gpu_layers=36,
        context_size=4096,
        grammar_input_variant="baseline",
        server_reasoning="auto",
        description=("同一 MoE 模型的 IQ4_XS 量化；保留为历史回归、量化和翻译盲评对照。"),
    ),
    LocalModelPreset(
        preset_id="qwen27_q4_k_s",
        display_name="Qwen3.5 27B · Q4_K_S（教学跨度覆盖候选）",
        model_alias="Qwen3.5-27B-Q4_K_S",
        filename="Qwen_Qwen3.5-27B-Q4_K_S.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="two-stage-span-id",
        server_reasoning="auto",
        description=(
            "span-ID 发现加通用教学跨度编辑；当前 reviewed 教学跨度覆盖领先，"
            "但尚未评估过宽范围、解释正确性或学习效果。"
        ),
    ),
    LocalModelPreset(
        preset_id="gemma4_26b_q4_0_it",
        display_name="Gemma 4 26B IT · Q4_0（低延迟候选）",
        model_alias="Gemma4-26B-IT-Q4_0",
        filename="gemma-4-26B_q4_0-it.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="span-id-balanced",
        server_reasoning="auto",
        description="当前低延迟候选配置；速度领先不代表教学质量领先。",
    ),
    LocalModelPreset(
        preset_id="gemma4_31b_q4_0_it",
        display_name="Gemma 4 31B IT · Q4_0（名称覆盖候选）",
        model_alias="Gemma4-31B-IT-Q4_0",
        filename="gemma-4-31B_q4_0-it.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="baseline",
        server_reasoning="auto",
        description=("当前 reviewed 目标名称覆盖领先；有限 rejected 清单为 0 不等于全量低误报。"),
    ),
    LocalModelPreset(
        preset_id="qwen38_27b_q4_k_m_dflash2",
        display_name="Qwen3.8 27B · Q4_K_M + DFlash2（默认加速）",
        model_alias="Qwen3.8-27B-Q4_K_M-DFlash2",
        filename="Qwen3.8-27B-Q4_K_M.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="compact-baseline",
        server_reasoning="off",
        description=(
            "当前默认性能配置；在同运行时配对测试中保持 reviewed 语法分，"
            "端到端 P50 降低约 35%。需要兼容 DFlash2 的 llama.cpp 运行时，"
            "不可用或启动失败时自动回退普通 Q4_K_M。"
        ),
        flash_attention="on",
        draft_filename="Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
        draft_gpu_layers=99,
        speculative_type="draft-dflash",
        draft_max_tokens=7,
        runtime_tag="pr27342",
    ),
    LocalModelPreset(
        preset_id="qwen38_27b_q4_k_m",
        display_name="Qwen3.8 27B · Q4_K_M（兼容回退）",
        model_alias="Qwen3.8-27B-Q4_K_M",
        filename="Qwen3.8-27B-Q4_K_M.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="compact-baseline",
        server_reasoning="off",
        description=(
            "不使用草稿模型的兼容配置；使用保留句面与词素证据的去重输入。"
            "显存压力低于 DFlash2，但解码和端到端延迟较高。"
        ),
    ),
    LocalModelPreset(
        preset_id="qwen38_27b_q5_k_m",
        display_name="Qwen3.8 27B · Q5_K_M（量化敏感性对照）",
        model_alias="Qwen3.8-27B-Q5_K_M",
        filename="Qwen3.8-27B-Q5_K_M.gguf",
        gpu_layers=99,
        context_size=4096,
        grammar_input_variant="baseline",
        server_reasoning="off",
        description=(
            "同基础模型的量化敏感性对照；额外显存和较低吞吐已观察到，"
            "当前格式敏感的小样本不能判定语法质量优劣。"
        ),
    ),
)

_PRESETS_BY_ID = {preset.preset_id: preset for preset in LOCAL_MODEL_PRESETS}


def local_model_preset(preset_id: str) -> LocalModelPreset | None:
    return _PRESETS_BY_ID.get(preset_id.strip())


def infer_local_model_preset(*, model: str, model_path: str) -> str:
    """Recover a preset from legacy settings without guessing from benchmark answers."""
    normalized_model = _normalized(model)
    normalized_path = _normalized(model_path)
    if not normalized_path and normalized_model == _normalized("Qwen3.5-35B-A3B"):
        return "qwen35_a3b_q3_k_xl"
    for preset in LOCAL_MODEL_PRESETS:
        if normalized_model and normalized_model == _normalized(preset.model_alias):
            return preset.preset_id
    path_matches = [
        preset
        for preset in LOCAL_MODEL_PRESETS
        if normalized_path and _normalized(preset.filename) in normalized_path
    ]
    if path_matches:
        # A main GGUF can serve both plain and speculative presets. A legacy path
        # does not prove that a draft model/runtime was configured, so recover the
        # non-speculative preset unless the saved alias selected DFlash explicitly.
        return next(
            (preset.preset_id for preset in path_matches if preset.speculative_type == "none"),
            path_matches[0].preset_id,
        )
    return CUSTOM_LOCAL_MODEL_PRESET


def _normalized(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())
