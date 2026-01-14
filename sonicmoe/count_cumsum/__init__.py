# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import torch

from ..enums import LIBRARY_NAME
from ..jit import cpp_jit
from .count_cumsum_triton import count_cumsum_triton


@torch.library.custom_op(f"{LIBRARY_NAME}::count_cumsum_cuda", mutates_args={"count_output", "cumsum_output"})
@cpp_jit()
def count_cumsum_cuda(x: torch.Tensor, count_output: torch.Tensor, cumsum_output: torch.Tensor | None) -> None: ...


@torch.no_grad()
def count_cumsum(x: torch.Tensor, E: int, do_cumsum: bool = True, use_triton: bool = True):
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]
    if use_triton:
        return count_cumsum_triton(x=x, E=E, do_cumsum=do_cumsum)

    count_output = torch.empty(E, dtype=torch.int32, device=x.device)
    cumsum_output = torch.empty(E, dtype=torch.int32, device=x.device) if do_cumsum else None

    if do_cumsum:
        return count_output, cumsum_output
    else:
        return count_output
