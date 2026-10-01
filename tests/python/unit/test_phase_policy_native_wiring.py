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
        self.cache_policy_calls = []
        self.cache_policy = 0

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

    def configure_expert_cache_policy(self, policy) -> None:
        if type(policy) is not int:
            raise TypeError("native cache policy must be an integer enum value")
        self.cache_policy_calls.append(policy)
        self.cache_policy = policy

    def get_expert_cache_policy(self):
        return self.cache_policy


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


def _config(enabled, monkeypatch, cache_policy="lru") -> ArcherConfig:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    return ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
        phase_specific_expert_policy=enabled,
        expert_cache_policy=cache_policy,
    )


def test_enabled_configures_handle_not_dispatcher(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == [
        (True, 1, 0, 1.0, 4.0, 8)
    ]
    assert engine.archer_engine.cache_policy_calls == [0]
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is True


def test_disabled_assigns_settings_but_does_not_configure(monkeypatch) -> None:
    engine = _bare_engine(_config(False, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == []
    assert engine.archer_engine.cache_policy_calls == []
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


@pytest.mark.parametrize(("policy", "expected"), [("lru", 0), ("lfu", 1)])
def test_cache_policy_follows_native_enum(monkeypatch, policy, expected) -> None:
    engine = _bare_engine(_config(True, monkeypatch, policy))

    engine._configure_native_phase_policy()

    assert engine.archer_engine.cache_policy_calls == [expected]
    assert engine.archer_engine.get_expert_cache_policy() == expected


def test_missing_native_cache_policy_api_fails_closed(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch, "lfu"))
    engine.archer_engine.configure_expert_cache_policy = None

    with pytest.raises(RuntimeError, match="native expert cache policy API"):
        engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == []


def test_native_cache_policy_readback_mismatch_fails_closed(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch, "lfu"))
    engine.archer_engine.get_expert_cache_policy = lambda: 0

    with pytest.raises(RuntimeError, match="readback mismatch"):
        engine._configure_native_phase_policy()
