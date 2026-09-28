import re

from moe_store.checkpoints import get_checkpoint_paths
from moe_store.parsing.gptq import is_gptq_packed_tensor, is_gptq_quantized
from moe_store.parsing.hf_config import (
    moe_text_config,
    parse_expert_dtype,
    parse_expert_id as _parse_expert_id,
    parse_moe_param as _parse_moe_param,
    resolve_config_dtype,
)
from moe_store.parsing.quantization import (
    QuantizationInfo,
    detect_quantization,
    validate_quantization_support,
)

from .async_transfer import async_d2h, async_h2d, wait_transfer
from .config import ArcherConfig
from .device import (
    DeviceConfig,
    get_default_device,
    get_device,
    get_num_devices,
    get_pinned_memory_device,
    is_cuda_available,
    to_device,
)


def _is_qwen2_moe(config) -> bool:
    architectures = getattr(config, "architectures", None) or [""]
    architecture = str(architectures[0]).lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    return "qwen2moe" in architecture or model_type == "qwen2_moe"


def _qwen2_sparse_layers(config) -> list[int]:
    num_layers = int(config.num_hidden_layers)
    num_experts = int(config.num_experts)
    sparse_step = int(getattr(config, "decoder_sparse_step", 1))
    if sparse_step <= 0:
        raise ValueError("Qwen2-MoE decoder_sparse_step must be positive")
    mlp_only = {
        int(layer) for layer in (getattr(config, "mlp_only_layers", None) or [])
    }
    if any(layer < 0 or layer >= num_layers for layer in mlp_only):
        raise ValueError("Qwen2-MoE mlp_only_layers contains an invalid layer")
    if num_experts <= 0:
        return []
    return [
        layer
        for layer in range(num_layers)
        if layer not in mlp_only and (layer + 1) % sparse_step == 0
    ]


def parse_moe_param(config):
    """Parse MoE dimensions, adding repository-local Qwen2-MoE support."""

    if not _is_qwen2_moe(config):
        return _parse_moe_param(config)
    return len(_qwen2_sparse_layers(config)), int(config.num_experts), 0


def parse_expert_id(param_name, config):
    """Map a Qwen2 checkpoint expert to the compact native layer index."""

    if not _is_qwen2_moe(config):
        return _parse_expert_id(param_name, config)
    match = re.search(
        r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
    )
    if match is None:
        return None, None
    raw_layer, expert_id = (int(value) for value in match.groups())
    if expert_id < 0 or expert_id >= int(config.num_experts):
        return None, None
    sparse_layers = _qwen2_sparse_layers(config)
    try:
        layer_id = sparse_layers.index(raw_layer)
    except ValueError:
        return None, None
    return layer_id, expert_id

__all__ = [
    "ArcherConfig",
    "async_d2h",
    "async_h2d",
    "detect_quantization",
    "DeviceConfig",
    "get_checkpoint_paths",
    "get_default_device",
    "get_device",
    "get_num_devices",
    "get_pinned_memory_device",
    "is_cuda_available",
    "is_gptq_packed_tensor",
    "is_gptq_quantized",
    "moe_text_config",
    "parse_expert_dtype",
    "parse_expert_id",
    "parse_moe_param",
    "resolve_config_dtype",
    "QuantizationInfo",
    "validate_quantization_support",
    "wait_transfer",
    "to_device",
]
