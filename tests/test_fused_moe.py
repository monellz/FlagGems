"""
Fused MoE correctness tests.

Tests FlagGems fused_experts_impl against:
  - vLLM's fused_experts_impl (bf16/fp16 and fp8_w8a8)
  - SonicMoE's moe_general_routing_inputs (bf16)
"""

from math import ceil
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
        # Qwen3.5-397B-A17B
        (1, 512, 4096, 1024, 10),
        (4, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        (128, 512, 4096, 1024, 10),
        (256, 512, 4096, 1024, 10),
    ]

# =====================================================================
# Helpers
# =====================================================================

def native_w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(compute_type)
    B = B.to(compute_type)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0], f"{n_tiles} == {Bs.shape[0]}"
    assert k_tiles == Bs.shape[1], f"{k_tiles} == {Bs.shape[1]}"

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=compute_type, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C

def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn
):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` must be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s

def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, block_shape):
    """Fused moe with block-wise quantization using native torch."""
    B, D = a.shape
    topk = topk_ids.size(1)
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    a_q = a_q.to(torch.float32)
    def silu_and_mul(x):
        import torch.nn.functional as F
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = silu_and_mul(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
            out[mask] = native_w8a8_block_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def _generate_moe_weights(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    block_shape: tuple[int, int] | None = None,
    sort_topk_ids: bool = False,
):
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


    w1 = (torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=torch.float32)
          * (1.0 / hidden_size ** 0.5))
    w1 = w1.to(dtype)
    if block_shape is not None:
        assert block_shape[0] == block_shape[1]
    if dtype == torch.float8_e4m3fn and block_shape is not None:
        assert (intermediate_size * 2) % block_shape[0] == 0, f"{(intermediate_size * 2)} % {block_shape[0]} != 0"
        assert hidden_size % block_shape[1] == 0, f"{hidden_size} % {block_shape[1]} != 0"
        w1_scale = torch.randn(
            num_experts,
            ceil(intermediate_size * 2 / block_shape[0]),
            ceil(hidden_size / block_shape[1]),
            device=device,
            dtype=torch.float32,
        )
    else:
        w1_scale = None

    w2 = (torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32)
          * (1.0 / intermediate_size ** 0.5))
    w2 = w2.to(dtype)
    if dtype == torch.float8_e4m3fn and block_shape is not None:
        assert block_shape[0] == block_shape[1]
        assert intermediate_size % block_shape[0] == 0, f"{intermediate_size} % {block_shape[0]} != 0"
        assert hidden_size % block_shape[1] == 0, f"{hidden_size} % {block_shape[1]} != 0"
        w2_scale = torch.randn(
            num_experts,
            ceil(hidden_size / block_shape[0]),
            ceil(intermediate_size / block_shape[1]),
            device=device,
            dtype=torch.float32,
        )
    else:
        w2_scale = None
    return w1, w2, w1_scale, w2_scale, topk_weights, topk_ids

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

try:
    import hpc
    HAS_HPC_FUSED_MOE = True
except ImportError:
    HAS_HPC_FUSED_MOE = False


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

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1, w2, _, _, topk_weights, topk_ids = _generate_moe_weights(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device, block_shape=None,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=num_experts,
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
# Tests: FlagGems vs torch ref (fp8 w8a8)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
def test_fused_moe_fp8_w8a8_blockwise_vs_torchref(config, block_shape):
    """FlagGems fused_moe (fp8 w8a8, block-wise) vs torch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    assert block_shape[0] == block_shape[1]
    if hidden_size % block_shape[1] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    if intermediate_size % block_shape[0] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1, w2, w1_scale, w2_scale, topk_weights, topk_ids = _generate_moe_weights(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, torch.float8_e4m3fn, device, block_shape=block_shape,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=num_experts,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_w8a8_block_fp8_moe(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, block_shape)
    torch.cuda.synchronize()

    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


# =====================================================================
# Tests: FlagGems vs vLLM (fp8 w8a8)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vllm not installed")
def test_fused_moe_fp8_w8a8_blockwise_vs_vllm(config, block_shape):
    """FlagGems fused_moe (fp8 w8a8, block-wise) vs torch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    assert block_shape[0] == block_shape[1]
    if hidden_size % block_shape[1] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    if intermediate_size % block_shape[0] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1, w2, w1_scale, w2_scale, topk_weights, topk_ids = _generate_moe_weights(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, torch.float8_e4m3fn, device, block_shape=block_shape,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=num_experts,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    ref = vllm_fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    torch.cuda.synchronize()

    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


# =====================================================================
# Tests: FlagGems vs hpc-ops (fp8 w8a8)
# =====================================================================

@pytest.mark.fused_moe
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
# @pytest.mark.parametrize("config",[
#     # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
#     (128, 128, 512, 512, 8),
# ])
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(not HAS_HPC_FUSED_MOE, reason="hpc-ops not installed")
def test_fused_moe_fp8_w8a8_blockwise_vs_hpc(config, block_shape):
    """FlagGems fused_moe (fp8 w8a8, block-wise) vs hpc-ops."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    assert block_shape[0] == block_shape[1]
    if block_shape[0] != 128 or block_shape[1] != 128:
        pytest.skip("Invalid block shape for hpc-ops")
    if hidden_size % block_shape[1] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    if intermediate_size % block_shape[0] != 0:
        pytest.skip("Invalid shape for block-wise quantization")
    if ceil(intermediate_size * 2 / 128) % 4 != 0:
        pytest.skip("Invalid shape for hpc-ops")
    if ceil(hidden_size / 128) % 4 != 0:
        pytest.skip("Invalid shape for hpc-ops")
    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1, w2, w1_scale, w2_scale, topk_weights, topk_ids = _generate_moe_weights(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, torch.float8_e4m3fn, device, block_shape=block_shape,
        sort_topk_ids=True,
    )
    from flag_gems.ops.per_token_group_quant_fp8 import per_token_group_quant_fp8
    hidden_states_q, a1_scale = per_token_group_quant_fp8(
        hidden_states,
        group_size=block_shape[1],
        dtype=torch.float8_e4m3fn,
        column_major_scales=False,
        scale_ue8m0=False,
    )
    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=num_experts,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    ref = hpc.fuse_moe_blockwise_fp8(
        hidden_states_q, 
        a1_scale,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        0,
        num_experts,
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

    # hidden_states, w1, w2, topk_weights, topk_ids = _generate_moe_inputs(
    #     num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device,
    # )
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1, w2, _, _, topk_weights, topk_ids = _generate_moe_weights(
        num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device, block_shape=None,
    )

    result = flag_gems.fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=num_experts,
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
