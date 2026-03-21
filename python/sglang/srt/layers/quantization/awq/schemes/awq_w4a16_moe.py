# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.quantization.awq_kernels import AWQAscendMoEKernel
from sglang.srt.layers.moe import MoeRunnerConfig

from .awq_moe import AWQMoEScheme

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.awq.awq import AWQConfig

__all__ = ["AWQAscendMoEScheme"]


class AWQAscendMoEScheme(AWQMoEScheme):
    def __init__(self, quant_config: "AWQConfig"):
        super().__init__(quant_config)
        self.kernel = AWQAscendMoEKernel(quant_config)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.kernel.moe_runner_config = moe_runner_config
