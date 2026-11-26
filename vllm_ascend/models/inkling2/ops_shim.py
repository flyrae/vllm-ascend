# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU stand-ins for upstream's ``nvidia/ops`` helpers used by the vendored
Inkling 2 modules."""


def use_sm100_kernels() -> bool:
    """NPU is never an SM100 device: the CUDA fast paths are never taken."""
    return False
