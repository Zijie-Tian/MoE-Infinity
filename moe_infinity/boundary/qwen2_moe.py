# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2-MoE adapter for the moe-store integration boundary.

Qwen2-MoE checkpoints store routed experts as individual
``gate_proj/up_proj/down_proj`` tensors, while transformers 5 builds a packed
expert module.  The native dispatcher consumes the individual tensors, so the
adapter preserves the checkpoint layout and delegates only the routed-expert
part of the block.  The shared expert remains ordinary PyTorch and therefore
continues through Archer's dense-layer offload lifecycle.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen2_moe.configuration_qwen2_moe import (
    Qwen2MoeConfig,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeForCausalLM,
    Qwen2MoeMLP,
    Qwen2MoeRotaryEmbedding,
    Qwen2MoeTopKRouter,
)


def ensure_qwen2_moe_registry() -> None:
    """Register Qwen2-MoE in the process-local moe-store registry.

    moe-store 0.2.1 does not publish this entry.  Updating its in-memory
    dictionaries keeps the compatibility shim repository-local; installed
    package files are never modified.
    """

    from moe_store.registry.constants import (
        MODEL_MAPPING_NAMES,
        MODEL_MAPPING_TYPES,
    )

    MODEL_MAPPING_NAMES.setdefault("qwen2moe", Qwen2MoeForCausalLM)
    # Qwen2 routed experts use the same SiLU(gate) * up -> down layout as the
    # native DeepSeek/Qwen3 family.
    MODEL_MAPPING_TYPES.setdefault("qwen2moe", 5)


class SyncQwen2MoeSparseMoeBlock(nn.Module):
    """Qwen2-MoE block whose routed experts execute via MoE-Infinity."""

    archer_config = None
    layer_id: Optional[int] = None

    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.gate = Qwen2MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            [
                Qwen2MoeMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = Qwen2MoeMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = nn.Linear(
            config.hidden_size, 1, bias=False
        )

        self.expert_executor = None
        self.expert_prefetcher = None
        self.expert_tracer = None
        self.expert_predictor = None
        self.archer_engine = None
        self.lib = None

    def _route(self, hidden_flat: torch.Tensor):
        router_logits, routing_weights, selected_experts = self.gate(
            hidden_flat
        )
        num_tokens = hidden_flat.shape[0]
        router_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=torch.bool,
            device=hidden_flat.device,
        )
        router_mask.scatter_(1, selected_experts, True)
        routing_weights_mask = torch.zeros(
            num_tokens,
            self.num_experts,
            dtype=routing_weights.dtype,
            device=hidden_flat.device,
        )
        routing_weights_mask.scatter_add_(
            1, selected_experts, routing_weights
        )
        return router_mask, routing_weights_mask, router_logits

    def _local_experts(
        self,
        hidden_flat: torch.Tensor,
        router_mask: torch.Tensor,
        routing_weights_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reference fallback used before the runtime wires an executor."""

        final_hidden = torch.zeros_like(hidden_flat)
        for expert_idx in range(self.num_experts):
            token_mask = router_mask[:, expert_idx]
            if not token_mask.any():
                continue
            expert_out = self.experts[expert_idx](hidden_flat[token_mask])
            weight = routing_weights_mask[
                token_mask, expert_idx
            ].unsqueeze(-1)
            final_hidden[token_mask] += (weight * expert_out).to(
                final_hidden.dtype
            )
        return final_hidden

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        router_mask, routing_weights_mask, router_logits = self._route(
            hidden_flat
        )
        if self.expert_executor is None:
            expert_output = self._local_experts(
                hidden_flat, router_mask, routing_weights_mask
            )
        else:
            self.expert_executor.dispatch_local(
                self.layer_id,
                hidden_flat,
                router_mask,
                routing_weights_mask,
                router_logits=router_logits,
            )
            expert_output = self.expert_executor.wait_dispatch_local()

        shared_output = self.shared_expert(hidden_flat)
        shared_output = (
            torch.sigmoid(self.shared_expert_gate(hidden_flat))
            * shared_output
        )
        expert_output = expert_output.reshape(-1, hidden_dim) + shared_output
        return expert_output.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)


class SyncQwen2MoeRotaryEmbedding(Qwen2MoeRotaryEmbedding):
    """Keep positional IDs beside Qwen2 hidden states before RoPE math.

    Qwen2 passes ``position_ids`` to the rotary module positionally.  With an
    offloaded model, HF may infer CPU for those IDs even though the embedding
    hook has already produced CUDA hidden states.  Aligning this one argument
    at the architecture boundary avoids broad argument-copying and leaves the
    upstream rotary implementation unchanged.
    """

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.device != x.device:
            position_ids = position_ids.to(x.device, non_blocking=True)
        return super().forward(x, position_ids)


__all__ = [
    "SyncQwen2MoeRotaryEmbedding",
    "SyncQwen2MoeSparseMoeBlock",
    "ensure_qwen2_moe_registry",
]
