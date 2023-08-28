'''
Implements fused mlp + gelu
'''

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_kernel(
    x_ptr, bias_ptr, residual_ptr,
    norm_w_ptr, norm_b_ptr,
    attn_output_ptr, norm_output_ptr,
    x_stride_0, x_stride_1,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * x_stride_0
    residual_ptr += row * x_stride_0
    attn_output_ptr += row * x_stride_0
    norm_output_ptr += row * x_stride_0

    # compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.)
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.)
        residual = tl.load(residual_ptr + cols, mask=cols < N, other=0.)
        attn_output = x + bias + residual
        tl.store(attn_output_ptr + cols, attn_output, mask=cols < N)

        _mean += attn_output.to(tl.float32)
    mean = tl.sum(_mean, axis=0) / N
    # compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        attn_output = tl.load(attn_output_ptr + cols, mask=cols < N, other=0.).to(tl.float32)
        attn_output = tl.where(cols < N, attn_output - mean, 0.)
        _var += attn_output * attn_output
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(norm_w_ptr + cols, mask=mask)
        b = tl.load(norm_b_ptr + cols, mask=mask)
        x = tl.load(attn_output_ptr + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(norm_output_ptr + cols, y, mask=mask)


def triton_fused_bias_residual_norm(x, residual, bias, norm_w, norm_b, eps=1e-5):
    # Check constraints.
    M, N = x.shape

    # Allocates output.
    attn_output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    norm_output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _triton_kernel[(M,)](
        x, bias, residual,
        norm_w, norm_b,
        attn_output, norm_output,
        x.stride(0), x.stride(1),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return attn_output, norm_output