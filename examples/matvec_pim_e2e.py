#!/usr/bin/env python3
"""
matvec_pim_e2e.py — Matrix-vector multiply with K-dimension reduction
for the Triton-IM → Ramulator2 e2e pipeline.

Implements  y = A @ x   where  A is [M, K]  and  x is [K].

The K-reduction is the *temporal loop* within each bank's PE: for each k,
the PE reads A[row, k] from bank memory (BR) and receives x[k] via
broadcast (W), then multiply-accumulates:  acc += A[row, k] * x[k].

This matches OptiPIM's scalar-MAC-per-cycle temporal reduction model.
No inter-bank reduction is needed because each bank computes complete
dot products for its own subset of output rows.

Usage::

    python scripts/run_triton_im_ramulator2_e2e.py examples/matvec_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/matvec_pim_e2e.py --skip-ramulator

Can also be run standalone for CPU verification::

    python examples/matvec_pim_e2e.py
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
def matvec_kernel(A, x, y, M, K, BLOCK: tl.constexpr):
    """y[i] = sum_k A[i, k] * x[k], tiled over BLOCK output rows.

    PIM mapping (per K-step):
      - A (STREAMED  → BR): bank-resident matrix, each PE reads its rows
      - x (OPERAND   → W):  input vector, x[k] broadcast to all PEs
      - y (ACCUMULATOR → BW): result vector, written once after K loop

    The for-k loop is the PIM temporal dimension (one MAC per PE per cycle).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in range(K):
        a_val = tl.load(A + offs * K + k, mask, other=0)   # A[row, k]  → BR
        x_val = tl.load(x + k)                              # x[k]       → W
        acc += a_val * x_val                                 # PE MAC
    tl.store(y + offs, acc, mask)                            # y[row]     → BW


# ── Configuration ─────────────────────────────────────────────────────

BLOCK     = 64
NUM_BANKS = 32
M         = 512    # output dimension (rows of A / length of y)
K         = 64     # reduction dimension (cols of A / length of x)


# ── PIM e2e config (required by run_triton_im_ramulator2_e2e.py) ─────

def pim_kernel_config():
    A = np.arange(1, M * K + 1, dtype=np.int32).reshape(M, K)
    x_vec = np.arange(1, K + 1, dtype=np.int32)
    y = np.zeros(M, dtype=np.int32)
    sig = {"A": "*i32", "x": "*i32", "y": "*i32",
           "M": "i32", "K": "i32", "BLOCK": "constexpr"}
    scalars = [M, K]
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
        "kernel":      matvec_kernel,
        "kernel_name": "matvec_kernel",
        "signature":   sig,
        "constants":   {"BLOCK": BLOCK},
        "attrs":       attrs,
        "num_banks":   NUM_BANKS,
        "grid":        (math.ceil(M / BLOCK),),          # 1D: 8 programs
        "tensors": [
            {"data": A.ravel(), "role": "streamed"},      # A — bank-resident matrix
            {"data": x_vec,     "role": "operand"},       # x — broadcast to all PEs
            {"data": y,         "role": "accumulator"},   # y — result vector
        ],
        "scalars": [M, K],
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
    lib = compile_im_kernel(llir, kernel_name="matvec_kernel")

    # Prepare data
    A_np = cfg["tensors"][0]["data"]
    x_np = cfg["tensors"][1]["data"]
    y_np = cfg["tensors"][2]["data"].copy()
    y_ref = A_np.reshape(M, K) @ x_np

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_c = A_np.ctypes.data_as(INT32_P)
    x_c = x_np.ctypes.data_as(INT32_P)
    y_c = y_np.ctypes.data_as(INT32_P)

    launch_im_kernel(lib, kernel_name="matvec_kernel",
                     num_banks=NUM_BANKS, num_programs=num_programs,
                     arg_types=[INT32_P, INT32_P, INT32_P,
                                ctypes.c_int32, ctypes.c_int32,
                                ctypes.c_void_p, ctypes.c_void_p],
                     arg_values=[A_c, x_c, y_c, M, K, None, None])

    if np.array_equal(y_np, y_ref):
        print(f"[PASS] All {M} output elements match (M={M}, K={K})")
        return 0
    mismatches = int(np.sum(y_np != y_ref))
    print(f"[FAIL] {mismatches}/{M} element(s) differ")
    return 1


if __name__ == "__main__":
    sys.exit(_verify())
