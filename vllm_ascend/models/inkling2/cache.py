# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV and convolution state of Inkling 2 for engine-free runs.

Upstream binds paged KV and conv-state caches from the vLLM engine. The NPU
attention layer here takes the state of one ``(layer, loop pass)`` explicitly
(``AscendInkling2LayerState``: a dense KV cache and the last inputs of its four
short-convolution streams). ``AscendInkling2DecodeCache`` holds one sequence's
states (batch size 1), so the model runs prefill, chunked prefill and decode
steps outside the engine. This module has no upstream counterpart: upstream
gets these tensors from the engine.
"""

import torch

from .layers import AscendInkling2DecoderLayer, AscendInkling2LayerState

__all__ = ["AscendInkling2DecodeCache", "AscendInkling2LayerState"]


class AscendInkling2DecodeCache:
    """States of one sequence (batch size 1) for engine-free runs.

    Created by ``AscendInkling2LoopModel.allocate_cache``. Every loop pass of a
    core layer has its own state, as upstream's per-pass KV and conv caches.
    """

    def __init__(
        self,
        layers: list[AscendInkling2DecoderLayer],
        loop_core: range,
        num_passes: int,
        max_length: int,
        device: torch.device | str,
    ) -> None:
        self.max_length = max_length
        self.length = 0
        self._states = {
            (i, p): layer.new_state(max_length, device)
            for i, layer in enumerate(layers)
            for p in range(num_passes if i in loop_core else 1)
        }

    def state(self, layer_idx: int, pass_idx: int) -> AscendInkling2LayerState:
        return self._states[(layer_idx, pass_idx)]

    def check_append(self, tokens: int, positions: torch.Tensor) -> None:
        """The ``tokens`` at ``positions`` must continue the sequence."""
        if self.length + tokens > self.max_length:
            raise ValueError(f"{self.length + tokens} tokens exceed the cache's max_length {self.max_length}")
        first, last = int(positions[0]), int(positions[-1])
        if (first, last) != (self.length, self.length + tokens - 1):
            raise ValueError(
                f"tokens at positions [{first}, {last}] do not continue the sequence at length {self.length}"
            )

    def reset(self) -> None:
        """Forget the sequence: zero every state."""
        self.length = 0
        for state in self._states.values():
            state.key.zero_()
            state.value.zero_()
            for conv in state.conv.values():
                conv.zero_()
