"""
Fused MoE performance benchmarks.

Benchmarks FlagGems fused_experts_impl against:
  - vLLM fused_experts_impl (bf16, fp8 w8a8 block-wise)
  - SonicMoE moe_general_routing_inputs (bf16)
"""

from math import ceil

import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark

# =====================================================================
# Shared shapes
# =====================================================================

MOE_SHAPES = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    # Mixtral-like
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    # DeepSeek-V3-like (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (128, 256, 7168, 2048, 8),
    (256, 256, 7168, 2048, 8),
]

SONICMOE_SHAPES = [
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
]

# =====================================================================
# Optional dependencies
# =====================================================================

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs
    HAS_SONICMOE = True
except ImportError:
    HAS_SONICMOE = False

try:
    import hpc
    HAS_HPC = True
except ImportError:
    HAS_HPC = False


# =====================================================================
# Input generators
# =====================================================================

def _generate_bf16_inputs(config, dtype, device):
    """Generate bf16/fp16 MoE inputs for benchmarking."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config

    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype,
    )
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype,
    ) * (1.0 / hidden_size ** 0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype,
    ) * (1.0 / intermediate_size ** 0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(gating, dim=-1), topk, dim=-1,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    return hidden_states, w1, w2, topk_weights, topk_ids


def _generate_fp8_blockwise_inputs(config, block_shape, device, sort_topk_ids=False):
    """Generate fp8 w8a8 block-wise quantized MoE inputs for benchmarking."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    block_n, block_k = block_shape
    dtype_bf16 = torch.bfloat16
    dtype_fp8 = torch.float8_e4m3fn

    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=dtype_bf16,
    )

    w1 = (torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=torch.bfloat16,
    ) * (1.0 / hidden_size ** 0.5)).to(dtype_fp8)
    w1_scale = torch.rand(
        num_experts,
        ceil(intermediate_size * 2 / block_n),
        ceil(hidden_size / block_k),
        device=device, dtype=torch.float32,
    ) + 0.01

    w2 = (torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16,
    ) * (1.0 / intermediate_size ** 0.5)).to(dtype_fp8)
    w2_scale = torch.rand(
        num_experts,
        ceil(hidden_size / block_n),
        ceil(intermediate_size / block_k),
        device=device, dtype=torch.float32,
    ) + 0.01

    if sort_topk_ids:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
        )
        topk_ids, _ = torch.sort(topk_ids, dim=1)
        topk_weights = torch.randn((num_tokens, topk), dtype=torch.float32, device=device) / topk
    else:
        gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating, dim=-1), topk, dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)

    return hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids


# =====================================================================
# Benchmark: FlagGems vs vLLM (bf16)
# =====================================================================

class FusedMoEBf16Benchmark(Benchmark):
    """fused_moe bf16: FlagGems vs vLLM."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = MOE_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._input_fn(config, cur_dtype)

    def _input_fn(self, config, dtype):
        inputs = _generate_bf16_inputs(config, dtype, flag_gems.device)
        yield inputs


def _vllm_bf16_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    return vllm_fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace=False, activation="silu",
    )


def _gems_bf16_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    return flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_bf16_gems_vs_vllm():
    """Benchmark FlagGems vs vLLM fused_moe (bf16)."""
    bench = FusedMoEBf16Benchmark(
        op_name="fused_moe_bf16_gems_vs_vllm",
        torch_op=_vllm_bf16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_bf16_wrapper)
    bench.run()


# =====================================================================
# Benchmark: FlagGems vs vLLM (fp8 w8a8 block-wise)
# =====================================================================

DEFAULT_BLOCK_SHAPE = [128, 128]


class FusedMoEFp8BlockwiseVLLMBenchmark(Benchmark):
    """fused_moe fp8 w8a8 block-wise: FlagGems vs vLLM."""

    def __init__(self, op_name, torch_op, dtypes, block_shape=None):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = block_shape or DEFAULT_BLOCK_SHAPE

    def set_shapes(self, shape_file_path=None):
        self.shapes = MOE_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            torch.cuda.empty_cache()
            yield from self._input_fn(config)

    def _input_fn(self, config):
        inputs = _generate_fp8_blockwise_inputs(
            config, self.block_shape, flag_gems.device,
        )
        yield inputs

@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_fp8_blockwise_gems_vs_vllm():
    """Benchmark FlagGems vs vLLM fused_moe (fp8 w8a8 block-wise 128x128)."""
    def _vllm_fp8_blockwise_wrapper(
        hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
    ):
        return vllm_fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids,
            inplace=False, activation="silu",
            use_fp8_w8a8=True,
            w1_scale=w1_scale, w2_scale=w2_scale,
            block_shape=DEFAULT_BLOCK_SHAPE,
        )
    def _gems_fp8_blockwise_wrapper(
        hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
    ):
        return flag_gems.fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids,
            use_fp8_w8a8=True,
            w1_scale=w1_scale, w2_scale=w2_scale,
            block_shape=DEFAULT_BLOCK_SHAPE,
        )
    bench = FusedMoEFp8BlockwiseVLLMBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_vllm",
        torch_op=_vllm_fp8_blockwise_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fp8_blockwise_wrapper)
    bench.run()


class FusedMoEFp8BlockwiseHPCBenchmark(Benchmark):
    """fused_moe fp8 w8a8 block-wise: FlagGems vs hpc-ops."""

    def __init__(self, op_name, torch_op, dtypes, block_shape=None):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = block_shape or DEFAULT_BLOCK_SHAPE

    def set_shapes(self, shape_file_path=None):
        self.shapes = MOE_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            torch.cuda.empty_cache()
            yield from self._input_fn(config)

    def _input_fn(self, config):
        num_experts = config[1]
        hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids = _generate_fp8_blockwise_inputs(
            config, self.block_shape, flag_gems.device,
            sort_topk_ids=True,
        )
        from flag_gems.ops.per_token_group_quant_fp8 import per_token_group_quant_fp8
        hidden_states, a1_scale = per_token_group_quant_fp8(
            hidden_states,
            group_size=self.block_shape[1],
            dtype=torch.float8_e4m3fn,
            column_major_scales=True,
            scale_ue8m0=False,
        )
        hidden_states = hidden_states.contiguous()
        a1_scale = a1_scale.contiguous()

        yield hidden_states, a1_scale, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_experts

@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_HPC, reason="hpc-ops not installed")
def test_perf_fused_moe_fp8_blockwise_gems_vs_hpc():
    """Benchmark FlagGems vs hpc-ops fused_moe (fp8 w8a8 block-wise 128x128)."""
    def _hpc_fp8_blockwise_wrapper(
        hidden_states, a1_scale, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_experts,
    ):
        return hpc.fuse_moe_blockwise_fp8(
            hidden_states, a1_scale, w1, w1_scale, w2, w2_scale, topk_ids, topk_weights, 0, num_experts,
        )
    def _gems_fp8_blockwise_wrapper(
        hidden_states, a1_scale, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_experts,
    ):
        return flag_gems.fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids,
            num_experts=num_experts,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=DEFAULT_BLOCK_SHAPE,
            a1_scale=a1_scale,
            out_dtype=torch.bfloat16, # must be bf16
        )
    bench = FusedMoEFp8BlockwiseHPCBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_hpc",
        torch_op=_hpc_fp8_blockwise_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fp8_blockwise_wrapper) 
    bench.run()



# =====================================================================
# Benchmark: FlagGems vs SonicMoE (bf16)
# =====================================================================

class FusedMoESonicmoeBenchmark(Benchmark):
    """fused_moe bf16: FlagGems vs SonicMoE."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = SONICMOE_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._input_fn(config, cur_dtype)

    def _input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states, w1, w2, topk_weights, topk_ids = _generate_bf16_inputs(
            config, dtype, device,
        )

        token_indices = (
            torch.arange(num_tokens, dtype=torch.int32, device=device)
            .unsqueeze(1).expand(-1, topk).reshape(-1)
        )
        expert_indices = topk_ids.reshape(-1)
        router_scores = topk_weights.reshape(-1)

        w1_sonic = torch.empty_like(w1)
        w1_sonic[:, 0::2, :] = w1[:, :intermediate_size, :]
        w1_sonic[:, 1::2, :] = w1[:, intermediate_size:, :]

        yield (
            hidden_states, w1, w2, topk_weights, topk_ids,
            token_indices, expert_indices, router_scores, w1_sonic,
        )


def _sonicmoe_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids,
    token_indices, expert_indices, router_scores, w1_sonic,
):
    num_experts = w1_sonic.shape[0]
    ref, _ = moe_general_routing_inputs(
        hidden_states,
        router_scores,
        token_indices,
        expert_indices,
        w1_sonic.permute(1, 2, 0),
        None,
        w2.permute(1, 2, 0),
        None,
        num_experts,
        torch.cuda.current_stream().cuda_stream,
        ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
    )
    return ref


def _gems_sonicmoe_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids,
    token_indices, expert_indices, router_scores, w1_sonic,
):
    return flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_SONICMOE, reason="sonicmoe not installed")
def test_perf_fused_moe_bf16_gems_vs_sonicmoe():
    """Benchmark FlagGems vs SonicMoE fused_moe (bf16)."""
    bench = FusedMoESonicmoeBenchmark(
        op_name="fused_moe_bf16_gems_vs_sonicmoe",
        torch_op=_sonicmoe_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_sonicmoe_wrapper)
    bench.run()
