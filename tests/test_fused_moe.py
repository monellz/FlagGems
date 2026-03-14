"""
Fused MoE correctness tests.

Tests FlagGems fused_experts_impl against:
  - vLLM's fused_experts_impl (bf16/fp16 and fp8_w8a8)
  - SonicMoE's moe_general_routing_inputs (bf16)
"""

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

# =====================================================================
# Shared test configurations
# =====================================================================

FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (4, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        (128, 8, 4096, 14336, 2),
        (256, 8, 4096, 14336, 2),
        (512, 8, 4096, 14336, 2),
        # DeepSeek-V3-like shapes (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (4, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        (128, 256, 7168, 2048, 8),
        (256, 256, 7168, 2048, 8),
    ]

# =====================================================================
# Helpers
# =====================================================================

def _generate_moe_inputs(num_tokens, num_experts, hidden_size,
                         intermediate_size, topk, dtype, device):
    """Generate shared MoE inputs: hidden_states, w1, w2, topk_weights, topk_ids."""
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


def _quantize_weights_fp8(w, dtype=torch.float8_e4m3fn):
    """Quantize a weight tensor to FP8 and return (w_fp8, w_scale).

    Per-tensor symmetric quantization: scale = max(|w|) / fp8_max.
    """
    finfo = torch.finfo(dtype)
    amax = w.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    scale = amax / finfo.max
    w_fp8 = (w / scale).clamp(finfo.min, finfo.max).to(dtype)
    # scale shape: [E, 1, 1] → squeeze to [E] for per-tensor
    return w_fp8, scale.squeeze(-1).squeeze(-1).to(torch.float32)


# =====================================================================
# Optional dependency imports
# =====================================================================

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )
    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False

try:
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs
    HAS_SONICMOE_FUSED_MOE = True
except ImportError:
    HAS_SONICMOE_FUSED_MOE = False


# =====================================================================
# Tests: FlagGems vs vLLM (bf16 / fp16)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_fused_moe_bf16_vs_vllm(config, dtype):
    """FlagGems fused_moe (bf16/fp16) vs vLLM reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    torch.manual_seed(0)

    hidden_states, w1, w2, topk_weights, topk_ids = _generate_moe_inputs(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        num_experts=num_experts,
    )

    ref = vllm_fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace=False, activation="silu",
    )

    torch.cuda.synchronize()

    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


# =====================================================================
# Tests: FlagGems vs vLLM (fp8 w8a8)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_fused_moe_fp8_w8a8_vs_vllm(config):
    """FlagGems fused_moe (fp8 w8a8, per-tensor) vs vLLM reference."""
    pytest.skip("fp8 w8a8 not yet implemented in fused_experts_impl")

    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states, w1, w2, topk_weights, topk_ids = _generate_moe_inputs(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device,
    )

    w1_fp8, w1_scale = _quantize_weights_fp8(w1)
    w2_fp8, w2_scale = _quantize_weights_fp8(w2)

    result = flag_gems.fused_experts_impl(
        hidden_states, w1_fp8, w2_fp8, topk_weights, topk_ids,
        num_experts=num_experts,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    ref = vllm_fused_experts_impl(
        hidden_states, w1_fp8, w2_fp8, topk_weights, topk_ids,
        inplace=False, activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    torch.cuda.synchronize()

    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


# =====================================================================
# Tests: FlagGems vs SonicMoE (bf16)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(not HAS_SONICMOE_FUSED_MOE, reason="sonicmoe not installed")
def test_fused_moe_vs_sonicmoe(config, dtype):
    """FlagGems fused_moe (bf16) vs SonicMoE reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    if hidden_size % 64 != 0 or hidden_size < 512 or intermediate_size % 64 != 0:
        pytest.skip("Invalid shape for SonicMoE")
    device = flag_gems.device
    torch.manual_seed(0)

    hidden_states, w1, w2, topk_weights, topk_ids = _generate_moe_inputs(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        num_experts=num_experts,
    )

    # SonicMoE expects interleaved gate/up layout and [N, K, E] weight order
    token_indices = (
        torch.arange(num_tokens, dtype=torch.int32, device=device)
        .unsqueeze(1).expand(-1, topk).reshape(-1)
    )
    expert_indices = topk_ids.reshape(-1)
    router_scores = topk_weights.reshape(-1)
    stream_id = torch.cuda.current_stream().cuda_stream

    w1_sonic = torch.empty_like(w1)
    w1_sonic[:, 0::2, :] = w1[:, :intermediate_size, :]
    w1_sonic[:, 1::2, :] = w1[:, intermediate_size:, :]

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
        stream_id,
        ActivationType.SWIGLU,
        is_inference_mode_enabled=True,
    )

    torch.cuda.synchronize()

    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)
