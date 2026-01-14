# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch
import triton
import triton.language as tl


@triton.jit
def _count_kernel(
    x_ptr,
    count_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load indices
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    # Atomic add
    tl.atomic_add(count_ptr + x, 1, mask=mask)


def count_cumsum_triton(x: torch.Tensor, E: int, do_cumsum: bool = True):
    """
    Triton implementation of count_cumsum.
    """
    assert x.is_cuda, "x must be on CUDA"
    assert x.dim() == 1, "x must be 1-dimensional"

    N = x.numel()

    # Ensure contiguous
    x = x.contiguous()

    count_output = torch.zeros(E, dtype=torch.int32, device=x.device)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    _count_kernel[grid](x, count_output, N, BLOCK_SIZE=1024)

    if do_cumsum:
        cumsum_output = torch.cumsum(count_output, dim=0, dtype=torch.int32)
        return count_output, cumsum_output

    return count_output
