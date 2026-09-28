from types import SimpleNamespace

import pytest
import torch
from moe_store.registry.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)
from transformers.models.qwen2_moe.configuration_qwen2_moe import (
    Qwen2MoeConfig,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeRotaryEmbedding,
    Qwen2MoeSparseMoeBlock,
)

from moe_infinity.boundary.qwen2_moe import (
    SyncQwen2MoeRotaryEmbedding,
    SyncQwen2MoeSparseMoeBlock,
    ensure_qwen2_moe_registry,
)
from moe_infinity.boundary.transformers_model_overrides import (
    install_transformers_model_overrides,
    restore_runtime_transformers_model_overrides,
)
from moe_infinity.utils import parse_expert_id, parse_moe_param


def _tiny_config() -> Qwen2MoeConfig:
    config = Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        decoder_sparse_step=1,
        hidden_act="silu",
        norm_topk_prob=False,
    )
    config.architectures = ["Qwen2MoeForCausalLM"]
    return config


def test_importing_moe_installs_qwen2_moe_registry_entry():
    from moe_infinity import MoE

    assert MoE.__name__ == "MoE"
    assert MODEL_MAPPING_NAMES["qwen2moe"].__name__ == (
        "Qwen2MoeForCausalLM"
    )
    assert MODEL_MAPPING_TYPES["qwen2moe"] == 5


def test_qwen2_moe_process_local_registration():
    ensure_qwen2_moe_registry()
    config = _tiny_config()

    assert MODEL_MAPPING_NAMES["qwen2moe"].__name__ == (
        "Qwen2MoeForCausalLM"
    )
    assert MODEL_MAPPING_TYPES["qwen2moe"] == 5
    assert parse_expert_type(config) == 5


def test_transformers_override_builds_sync_qwen2_blocks():
    from transformers.models.qwen2_moe.modeling_qwen2_moe import (
        Qwen2MoeForCausalLM,
    )

    try:
        install_transformers_model_overrides()
        model = Qwen2MoeForCausalLM(_tiny_config())
        blocks = [
            module
            for module in model.modules()
            if isinstance(module, SyncQwen2MoeSparseMoeBlock)
        ]
        assert len(blocks) == 2
        assert isinstance(
            model.model.rotary_emb, SyncQwen2MoeRotaryEmbedding
        )
    finally:
        restore_runtime_transformers_model_overrides()


def test_qwen2_rotary_aligns_positional_ids_to_hidden_device():
    rotary = SyncQwen2MoeRotaryEmbedding(_tiny_config())
    hidden_states = torch.empty((1, 2, 3, 8), device="meta")
    position_ids = torch.arange(3).unsqueeze(0)

    cos, sin = rotary(hidden_states, position_ids)

    assert cos.device.type == "meta"
    assert sin.device.type == "meta"
    assert cos.shape == (1, 3, 8)
    assert sin.shape == (1, 3, 8)


def test_qwen2_rotary_preserves_upstream_math_on_same_device():
    config = _tiny_config()
    upstream = Qwen2MoeRotaryEmbedding(config)
    compatible = SyncQwen2MoeRotaryEmbedding(config)
    compatible.load_state_dict(upstream.state_dict())
    hidden_states = torch.randn(1, 2, 3, 8)
    position_ids = torch.arange(3).unsqueeze(0)

    expected = upstream(hidden_states, position_ids)
    actual = compatible(hidden_states, position_ids)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_qwen2_moe_parser_maps_sparse_layers_to_compact_ids():
    config = SimpleNamespace(
        architectures=["Qwen2MoeForCausalLM"],
        model_type="qwen2_moe",
        num_hidden_layers=6,
        num_experts=4,
        decoder_sparse_step=2,
        mlp_only_layers=[3],
    )

    assert parse_moe_param(config) == (2, 4, 0)
    assert parse_expert_id(
        "model.layers.1.mlp.experts.3.gate_proj.weight", config
    ) == (0, 3)
    assert parse_expert_id(
        "model.layers.5.mlp.experts.0.down_proj.weight", config
    ) == (1, 0)
    # Layer 3 would satisfy sparse_step, but mlp_only_layers makes it dense.
    assert parse_expert_id(
        "model.layers.3.mlp.experts.0.up_proj.weight", config
    ) == (None, None)
    assert parse_expert_id(
        "model.layers.1.mlp.shared_expert.gate_proj.weight", config
    ) == (None, None)


def test_qwen2_moe_parser_rejects_invalid_sparse_shape():
    config = SimpleNamespace(
        architectures=["Qwen2MoeForCausalLM"],
        model_type="qwen2_moe",
        num_hidden_layers=2,
        num_experts=4,
        decoder_sparse_step=0,
        mlp_only_layers=[],
    )

    with pytest.raises(ValueError, match="decoder_sparse_step"):
        parse_moe_param(config)


def test_qwen2_moe_parser_rejects_out_of_range_expert():
    config = _tiny_config()
    assert parse_expert_id(
        "model.layers.0.mlp.experts.4.gate_proj.weight", config
    ) == (None, None)


def test_sync_qwen2_block_matches_hf_gate_experts_and_shared_path():
    config = _tiny_config()
    torch.manual_seed(7)

    hf_block = Qwen2MoeSparseMoeBlock(config).eval()
    for parameter in hf_block.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)

    sync_block = SyncQwen2MoeSparseMoeBlock(config).eval()
    state = dict(hf_block.state_dict())
    gate_up = state.pop("experts.gate_up_proj")
    down = state.pop("experts.down_proj")
    for expert_id in range(config.num_experts):
        gate, up = gate_up[expert_id].chunk(2, dim=0)
        prefix = f"experts.{expert_id}"
        state[f"{prefix}.gate_proj.weight"] = gate.contiguous()
        state[f"{prefix}.up_proj.weight"] = up.contiguous()
        state[f"{prefix}.down_proj.weight"] = down[
            expert_id
        ].contiguous()

    missing, unexpected = sync_block.load_state_dict(state, strict=False)
    assert missing == []
    assert unexpected == []

    hidden_states = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        expected = hf_block(hidden_states)
        actual = sync_block(hidden_states)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-5)


def test_sync_qwen2_block_keeps_shared_expert_outside_executor():
    config = _tiny_config()
    block = SyncQwen2MoeSparseMoeBlock(config)

    routed_names = {
        name for name, _ in block.named_parameters() if name.startswith("experts.")
    }
    shared_names = {
        name
        for name, _ in block.named_parameters()
        if name.startswith("shared_expert")
    }
    assert routed_names
    assert shared_names
    assert routed_names.isdisjoint(shared_names)
    assert block.expert_executor is None
