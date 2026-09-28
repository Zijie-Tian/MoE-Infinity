from types import SimpleNamespace

import pytest
from transformers import MixtralConfig, MixtralForCausalLM

from moe_infinity.boundary.transformers_model_overrides import (
    install_transformers_model_overrides,
    restore_context_transformers_model_overrides,
    restore_runtime_transformers_model_overrides,
)
from moe_infinity.runtime.model_offload import (
    OffloadEngine,
    _add_mixtral_runtime_name_aliases,
    _make_expert_nbytes_map,
    _make_expert_tensor_map,
    _mixtral_checkpoint_name,
)
from moe_store.wrappers import SyncMixtralSparseMoeBlock


def _tiny_mixtral():
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        architectures=["MixtralForCausalLM"],
    )
    installed = False
    try:
        install_transformers_model_overrides()
        installed = True
        model = MixtralForCausalLM(config)
    finally:
        if installed:
            restore_context_transformers_model_overrides()
            restore_runtime_transformers_model_overrides()
    return model, config


def _canonical_name_id_map(model, config):
    return {
        _mixtral_checkpoint_name(name, config): tensor_id
        for tensor_id, (name, _) in enumerate(
            model.named_parameters(recurse=True), start=1
        )
    }


def test_mixtral_runtime_aliases_build_complete_ordered_sparse_topology():
    model, config = _tiny_mixtral()
    assert all(
        isinstance(layer.mlp, SyncMixtralSparseMoeBlock)
        for layer in model.model.layers
    )
    name_id_map = _canonical_name_id_map(model, config)
    canonical_names = set(name_id_map)

    added = _add_mixtral_runtime_name_aliases(name_id_map, model, config)

    runtime_names = [name for name, _ in model.named_parameters(recurse=True)]
    assert added > 0
    assert all(name in name_id_map for name in runtime_names)
    assert canonical_names <= set(name_id_map)
    for name, param in model.named_parameters(recurse=True):
        param.ar_id = name_id_map.get(name)
        assert param.ar_id is not None

    expert_map = _make_expert_tensor_map(name_id_map, config)
    assert set(expert_map) == {
        (layer_id, expert_id)
        for layer_id in range(2)
        for expert_id in range(3)
    }
    assert set(_make_expert_nbytes_map(model, config)) == set(expert_map)

    engine = object.__new__(OffloadEngine)
    engine.config = config
    engine.name_id_map = name_id_map
    topology = engine.get_topology(model)
    sparse = [groups for _, groups in topology if len(groups) > 1]
    assert len(sparse) == 2
    for layer_id, experts in enumerate(sparse):
        assert len(experts) == 3
        for expert_id, tensor_ids in enumerate(experts):
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            assert tensor_ids == [
                name_id_map[f"{prefix}.w1.weight"],
                name_id_map[f"{prefix}.w2.weight"],
                name_id_map[f"{prefix}.w3.weight"],
            ]


def test_mixtral_alias_rejects_conflicting_runtime_id():
    model, config = _tiny_mixtral()
    name_id_map = _canonical_name_id_map(model, config)
    runtime_name = "model.layers.0.mlp.experts.0.w1.weight"
    canonical_name = _mixtral_checkpoint_name(runtime_name, config)
    name_id_map[runtime_name] = name_id_map[canonical_name] + 1

    with pytest.raises(RuntimeError, match="Conflicting Mixtral"):
        _add_mixtral_runtime_name_aliases(name_id_map, model, config)


def test_non_mixtral_does_not_add_aliases():
    model, _ = _tiny_mixtral()
    config = SimpleNamespace(model_type="deepseek_v2")
    name_id_map = {"sentinel": 7}

    assert _add_mixtral_runtime_name_aliases(name_id_map, model, config) == 0
    assert name_id_map == {"sentinel": 7}
    assert _mixtral_checkpoint_name("model.layers.0.mlp.weight", config) == (
        "model.layers.0.mlp.weight"
    )


def test_mixtral_setup_rejects_incomplete_sparse_topology():
    engine = object.__new__(OffloadEngine)
    engine.config = SimpleNamespace(model_type="mixtral", num_hidden_layers=2)
    engine.name_id_map = {}
    engine.archer_engine = SimpleNamespace(register=lambda *_: None)
    engine.get_topology = lambda _model: [("model.layers.0", [[1]])]
    model = SimpleNamespace(
        named_parameters=lambda recurse=True: [],
        named_buffers=lambda recurse=True: [],
    )

    with pytest.raises(RuntimeError, match="expected 2 sparse layers, got 0"):
        engine.setup_archer_hooks(model)
