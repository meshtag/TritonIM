#!/usr/bin/env python3
"""
matmul_pim_e2e.py — Matrix multiply with K-dimension reduction
for the Triton-IM → Ramulator2 e2e pipeline.

Implements  C = A @ B   where  A is [M, K],  B is [K, N],  C is [M, N].

The M*N output elements are flattened into a 1D BLOCK and distributed
across PIM banks.  Each bank's PE computes complete dot products for its
assigned output elements via the temporal K-loop:

    for each k:
        BR — read  A[row, k]  from bank memory
        W  — load  B[k, col]  to PE register
        MAC — acc += A[row, k] * B[k, col]

No inter-bank reduction is needed because each bank owns disjoint (row, col)
output elements and runs the full K accumulation locally.

Usage::

    python scripts/run_triton_im_ramulator2_e2e.py examples/matmul_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/matmul_pim_e2e.py --skip-ramulator

Can also be run standalone for CPU verification::

    python examples/matmul_pim_e2e.py
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
def matmul_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
    """C[i,j] = sum_k A[i,k] * B[k,j], flattened over M*N outputs.

    PIM mapping (per K-step):
      - A (STREAMED  → BR): bank-resident weights, each PE reads its rows
      - B (OPERAND   → W):  input matrix, B[k, col] loaded to PE
      - C (ACCUMULATOR → BW): result matrix, written once after K loop

    Output elements are linearised: flat index → (row, col) via div/mod.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N

    rows = offs // N
    cols = offs % N

    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in range(K):
        a_val = tl.load(A + rows * K + k, mask, other=0)   # A[row, k] → BR
        b_val = tl.load(B + k * N + cols, mask, other=0)   # B[k, col] → W
        acc += a_val * b_val                                 # PE MAC
    tl.store(C + offs, acc, mask)                            # C[i,j]   → BW


# ── Configuration ─────────────────────────────────────────────────────

BLOCK     = 64
NUM_BANKS = 32
M         = 16     # rows of A / rows of C
N         = 64     # cols of B / cols of C
K         = 32     # reduction dimension


# ── PIM e2e config (required by run_triton_im_ramulator2_e2e.py) ─────

def pim_kernel_config():
    A = np.arange(1, M * K + 1, dtype=np.int32).reshape(M, K)
    B = np.arange(1, K * N + 1, dtype=np.int32).reshape(K, N)
    C = np.zeros((M, N), dtype=np.int32)
    sig = {"A": "*i32", "B": "*i32", "C": "*i32",
           "M": "i32", "N": "i32", "K": "i32",
           "BLOCK": "constexpr"}
    scalars = [M, N, K]
    attrs = {}
    si = 0
    for i, ty in enumerate(v for v in sig.values() if v != "constexpr"):
        if ty.startswith("*"):
            attrs[(i,)] = [["tt.divisibility", 16]]
        else:
            if scalars[si] % 16 == 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
            si += 1
    return {
        "kernel":      matmul_kernel,
        "kernel_name": "matmul_kernel",
        "signature":   sig,
        "constants":   {"BLOCK": BLOCK},
        "attrs":       attrs,
        "num_banks":   NUM_BANKS,
        "grid":        (math.ceil(M * N / BLOCK),),      # 1D: 16 programs
        "tensors": [
            {"data": A.ravel(), "role": "streamed"},      # A — bank-resident weights
            {"data": B.ravel(), "role": "operand"},       # B — input, loaded to PE
            {"data": C.ravel(), "role": "accumulator"},   # C — result matrix
        ],
        "scalars": [M, N, K],
    }


# ── Standalone CPU verification ──────────────────────────────────────

def _verify() -> int:
    """Compile via IM backend, run on CPU, compare with NumPy."""
    import ctypes
    from triton.backends.im import IMTarget, compile_im_kernel, launch_im_kernel
    from triton.compiler import ASTSource, make_backend

    cfg = pim_kernel_config()
    num_programs = cfg["grid"][0]

    # Compile
    src = ASTSource(cfg["kernel"], cfg["signature"],
                    constexprs=cfg["constants"], attrs=cfg["attrs"])
    target = IMTarget("hbm-pim", NUM_BANKS)
    backend = make_backend(target)
    opts = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=target, options=opts.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")
    lib = compile_im_kernel(llir, kernel_name="matmul_kernel")

    # Prepare data
    A_np = cfg["tensors"][0]["data"].reshape(M, K)
    B_np = cfg["tensors"][1]["data"].reshape(K, N)
    C_np = cfg["tensors"][2]["data"].copy()
    C_ref = (A_np @ B_np).ravel()

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_c = cfg["tensors"][0]["data"].ctypes.data_as(INT32_P)
    B_c = cfg["tensors"][1]["data"].ctypes.data_as(INT32_P)
    C_c = C_np.ctypes.data_as(INT32_P)

    launch_im_kernel(lib, kernel_name="matmul_kernel",
                     num_banks=NUM_BANKS, num_programs=num_programs,
                     arg_types=[INT32_P, INT32_P, INT32_P,
                                ctypes.c_int32, ctypes.c_int32,
                                ctypes.c_int32,
                                ctypes.c_void_p, ctypes.c_void_p],
                     arg_values=[A_c, B_c, C_c, M, N, K, None, None])

    if np.array_equal(C_np, C_ref):
        print(f"[PASS] All {M}×{N} = {M * N} output elements match (K={K})")
        return 0
    mismatches = int(np.sum(C_np != C_ref))
    print(f"[FAIL] {mismatches}/{M * N} element(s) differ")
    return 1


if __name__ == "__main__":
    sys.exit(_verify())
