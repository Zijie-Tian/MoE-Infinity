# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

import pytest

from moe_infinity.runtime.model_offload import OffloadEngine
from moe_infinity.utils import ArcherConfig


class _RecordingHandle:
    def __init__(self) -> None:
        self.calls = []

    def configure_phase_policy(
        self,
        enabled,
        prefill_admission,
        decode_admission,
        prefill_weight,
        decode_weight,
        starvation_limit,
    ) -> None:
        if (
            type(prefill_admission) is not int
            or type(decode_admission) is not int
        ):
            raise TypeError("native admission modes must be integer enum values")
        self.calls.append(
            (
                enabled,
                prefill_admission,
                decode_admission,
                prefill_weight,
                decode_weight,
                starvation_limit,
            )
        )


class _RecordingDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def configure_phase_policy(self, *args) -> None:
        self.calls.append(args)


def _bare_engine(archer_config) -> OffloadEngine:
    engine = object.__new__(OffloadEngine)
    engine.archer_config = archer_config
    engine.archer_engine = _RecordingHandle()
    engine.expert_dispatcher = _RecordingDispatcher()
    engine.expert_prefetcher = SimpleNamespace(phase_policy=None)
    return engine


def _config(enabled, monkeypatch) -> ArcherConfig:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    return ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
        phase_specific_expert_policy=enabled,
    )


def test_enabled_configures_handle_not_dispatcher(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == [
        (True, 1, 0, 1.0, 4.0, 8)
    ]
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is True


def test_disabled_assigns_settings_but_does_not_configure(monkeypatch) -> None:
    engine = _bare_engine(_config(False, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == []
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is False


@pytest.mark.parametrize(
    ("prefill", "decode", "expected"),
    [
        ("cache", "cache", (0, 0)),
        ("cache", "transient_on_pressure", (0, 1)),
        ("transient_on_pressure", "transient_on_pressure", (1, 1)),
    ],
)
def test_admission_modes_follow_native_enum(
    monkeypatch, prefill, decode, expected
) -> None:
    config = _config(True, monkeypatch)
    config.prefill_expert_admission = prefill
    config.decode_expert_admission = decode
    engine = _bare_engine(config)

    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls[0][1:3] == expected
    assert engine.expert_prefetcher.phase_policy.prefill_admission == prefill
    assert engine.expert_prefetcher.phase_policy.decode_admission == decode
