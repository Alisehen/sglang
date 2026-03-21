from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    npu_fused_experts,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

import torch_npu


class AWQAscendLinearKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
        qweight_tmp = torch.zeros_like(layer.qweight.data)
        qzeros_tmp = layer.qzeros.data
        qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        for i in range(0, self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            qzeros_list.append((qzeros_tmp.reshape(-1, 1) >> shift_num) & 0xF)
            qweight_tmp.bitwise_or_(
                ((layer.qweight.data >> shift_num) * (2 ** (4 * i))) & (0xF << (4 * i))
            )

        qweight_tmp.bitwise_xor_(0x88888888)

        qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(qzeros_tmp.shape[0], -1)
        qzeros_tmp = -(qzeros_tmp - 8)
        qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)

        layer.zeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)
        layer.weight = torch.nn.Parameter(qweight_tmp, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.weight
        scales = layer.scales
        qzeros = layer.zeros
        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        out = torch_npu.npu_weight_quant_batchmatmul(
            reshaped_x,
            qweight,
            antiquant_scale=scales,
            antiquant_offset=qzeros,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
        )

        return out.reshape(out_shape)


class AWQAscendMoEKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config
        self.moe_runner_config: Optional["MoeRunnerConfig"] = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_qweight_tmp = torch.zeros_like(layer.w13_qweight.data)
        w2_qweight_tmp = torch.zeros_like(layer.w2_qweight.data)
        w13_qzeros_list = []
        w2_qzeros_list = []
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]
        for i in range(0, self.quant_config.pack_factor):
            shift_num = shifts[i] * 4
            w13_qzeros_list.append(
                (layer.w13_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w2_qzeros_list.append(
                (layer.w2_qzeros.data.reshape(-1, 1) >> shift_num) & 0xF
            )
            w13_qweight_tmp.bitwise_or_(
                ((layer.w13_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )
            w2_qweight_tmp.bitwise_or_(
                ((layer.w2_qweight.data >> shift_num) * (2 ** (4 * i)))
                & (0xF << (4 * i))
            )

        w13_qweight_tmp.bitwise_xor_(0x88888888)
        w2_qweight_tmp.bitwise_xor_(0x88888888)

        w13_qzeros_tmp = torch.cat(w13_qzeros_list, dim=-1).reshape(
            layer.w13_qzeros.shape[0], layer.w13_qzeros.shape[1], -1
        )
        w13_qzeros_tmp = -(w13_qzeros_tmp - 8)
        w13_qzeros_tmp = w13_qzeros_tmp.to(layer.w13_scales.data.dtype)
        w2_qzeros_tmp = torch.cat(w2_qzeros_list, dim=-1).reshape(
            layer.w2_qzeros.shape[0], layer.w2_qzeros.shape[1], -1
        )
        w2_qzeros_tmp = -(w2_qzeros_tmp - 8)
        w2_qzeros_tmp = w2_qzeros_tmp.to(layer.w2_scales.data.dtype)

        layer.register_parameter(
            "w13_qzeros", torch.nn.Parameter(w13_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w13_qweight", torch.nn.Parameter(w13_qweight_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qzeros", torch.nn.Parameter(w2_qzeros_tmp, requires_grad=False)
        )
        layer.register_parameter(
            "w2_qweight", torch.nn.Parameter(w2_qweight_tmp, requires_grad=False)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config is not None, "moe runner is not initialized"
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)
        output = npu_fused_experts(
            hidden_states=x,
            w13=layer.w13_qweight,
            w13_scale=layer.w13_scales,
            w13_offset=layer.w13_qzeros,
            w2=layer.w2_qweight,
            w2_scale=layer.w2_scales,
            w2_offset=layer.w2_qzeros,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
            use_wna16=True,
        )
        return StandardCombineInput(hidden_states=output)
