import torch

from moe_infinity.runtime.model_offload import OffloadEngine


class PositionProbe(torch.nn.Module):
    def forward(self, hidden_states=None, *, position_ids=None, marker=None):
        if position_ids is not None:
            assert position_ids.device == hidden_states.device
        return position_ids, marker


def attach_runtime_hooks(module):
    engine = object.__new__(OffloadEngine)
    engine.forward_hooks = []
    engine.offload_set = set()
    engine._enable_kv_cache_offload = False
    engine._register_hooks_recursively(module, count=[0])
    return engine


def test_keyword_positions_follow_positional_hidden_without_moving_other_data():
    module = PositionProbe()
    engine = attach_runtime_hooks(module)
    hidden = torch.empty(1, 4, device="meta")
    positions = torch.arange(4).unsqueeze(0)
    marker = torch.ones(1)

    output, returned_marker = module(
        hidden, position_ids=positions, marker=marker
    )

    assert output.device.type == "meta"
    assert positions.device.type == "cpu"
    assert returned_marker is marker
    assert engine.forward_hooks


def test_keyword_positions_follow_keyword_hidden():
    module = PositionProbe()
    attach_runtime_hooks(module)
    output, _ = module(
        hidden_states=torch.empty(1, 4, device="meta"),
        position_ids=torch.arange(4).unsqueeze(0),
    )
    assert output.device.type == "meta"


def test_same_device_positions_are_not_copied():
    module = PositionProbe()
    attach_runtime_hooks(module)
    positions = torch.arange(4).unsqueeze(0)
    output, _ = module(torch.zeros(1, 4), position_ids=positions)
    assert output is positions


def test_missing_positions_are_unchanged():
    module = PositionProbe()
    attach_runtime_hooks(module)
    assert module(torch.empty(1, 4, device="meta")) == (None, None)
