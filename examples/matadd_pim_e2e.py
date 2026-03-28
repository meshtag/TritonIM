#!/usr/bin/env python3
"""
matadd_pim_e2e.py — 2D MatAdd kernel config for the Triton-IM → Ramulator2 e2e pipeline.

Usage::

    python scripts/run_triton_im_ramulator2_e2e.py examples/matadd_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/matadd_pim_e2e.py --skip-ramulator

Can also be run standalone for CPU verification::

    python examples/matadd_pim_e2e.py
"""

import math
import os
import sys

import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
import triton.language as tl


# ── Kernel ────────────────────────────────────────────────────────────

@triton.jit
def matadd_kernel(A, B, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """C[i, j] = A[i, j] + B[i, j], tiled in 2D."""
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offs = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)

    offs = row_offs[:, None] * N + col_offs[None, :]
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)

    a = tl.load(A + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    tl.store(C + offs, a + b, mask)


# ── Configuration ─────────────────────────────────────────────────────

BLOCK_M   = 4
BLOCK_N   = 64
NUM_BANKS = 32
M         = 16
N_SIZE    = 256


# ── PIM e2e config (required by run_triton_im_ramulator2_e2e.py) ─────

def pim_kernel_config():
    A = np.arange(1, M * N_SIZE + 1, dtype=np.int32).reshape(M, N_SIZE)
    B = np.arange(10, 10 * (M * N_SIZE) + 1, 10, dtype=np.int32).reshape(M, N_SIZE)
    C = np.zeros((M, N_SIZE), dtype=np.int32)
    return {
        "kernel":      matadd_kernel,
        "kernel_name": "matadd_kernel",
        "signature":   {"A": "*i32", "B": "*i32", "C": "*i32",
                        "M": "i32", "N": "i32",
                        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        "constants":   {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        "num_banks":   NUM_BANKS,
        "grid":        (math.ceil(M / BLOCK_M), math.ceil(N_SIZE / BLOCK_N)),  # 2D: (4, 4)
        "tensors": [
            {"data": A.ravel(), "role": "streamed"},       # A — bank-resident
            {"data": B.ravel(), "role": "operand"},        # B — loaded to PE register
            {"data": C.ravel(), "role": "accumulator"},    # C — result
        ],
        "scalars": [M, N_SIZE],
    }


# ── Standalone CPU verification ──────────────────────────────────────

def _verify() -> int:
    import ctypes
    from triton.backends.im import IMTarget, compile_im_kernel, launch_im_kernel
    from triton.compiler import ASTSource, make_backend

    cfg = pim_kernel_config()
    nx, ny = cfg["grid"]

    src = ASTSource(cfg["kernel"], cfg["signature"],
                    constexprs=cfg["constants"], attrs={})
    target = IMTarget("hbm-pim", NUM_BANKS)
    backend = make_backend(target)
    opts = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=target, options=opts.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")
    lib = compile_im_kernel(llir, kernel_name="matadd_kernel")

    A_np = cfg["tensors"][0]["data"]
    B_np = cfg["tensors"][1]["data"]
    C_np = cfg["tensors"][2]["data"].copy()
    C_ref = A_np + B_np

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_c = A_np.ctypes.data_as(INT32_P)
    B_c = B_np.ctypes.data_as(INT32_P)
    C_c = C_np.ctypes.data_as(INT32_P)

    launch_im_kernel(lib, kernel_name="matadd_kernel",
                     num_banks=NUM_BANKS, num_programs=nx,
                     num_programs_y=ny,
                     arg_types=[INT32_P, INT32_P, INT32_P,
                                ctypes.c_int32, ctypes.c_int32,
                                ctypes.c_void_p, ctypes.c_void_p],
                     arg_values=[A_c, B_c, C_c, M, N_SIZE, None, None])

    if np.array_equal(C_np, C_ref):
        print(f"[PASS] All {M}×{N_SIZE} = {M * N_SIZE} elements match")
        return 0
    mismatches = int(np.sum(C_np != C_ref))
    print(f"[FAIL] {mismatches} element(s) differ")
    return 1


if __name__ == "__main__":
    sys.exit(_verify())
