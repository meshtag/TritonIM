#!/usr/bin/env python3
"""
axpy_pim_e2e.py — AXPY kernel config for the Triton-IM → Ramulator2 e2e pipeline.

Usage::

    python scripts/run_triton_im_ramulator2_e2e.py examples/axpy_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/axpy_pim_e2e.py --skip-ramulator

Can also be run standalone for CPU verification::

    python examples/axpy_pim_e2e.py
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
def axpy_kernel(X, Y, A, N, BLOCK: tl.constexpr):
    """Y[i] = A * X[i] + Y[i], tiled over BLOCK elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask, other=0)
    y = tl.load(Y + offs, mask, other=0)
    out = A * x + y
    tl.store(Y + offs, out, mask)


# ── Configuration ─────────────────────────────────────────────────────

BLOCK     = 64
NUM_BANKS = 32
N         = 512
A_SCALAR  = 3


# ── PIM e2e config (required by run_triton_im_ramulator2_e2e.py) ─────

def pim_kernel_config():
    X = np.arange(1, N + 1, dtype=np.int32)
    Y = np.arange(10, 10 * N + 1, 10, dtype=np.int32)
    return {
        "kernel":      axpy_kernel,
        "kernel_name": "axpy_kernel",
        "signature":   {"X": "*i32", "Y": "*i32", "A": "i32", "N": "i32",
                        "BLOCK": "constexpr"},
        "constants":   {"BLOCK": BLOCK},
        "num_banks":   NUM_BANKS,
        "grid":        (math.ceil(N / BLOCK),),          # 1D: 8 programs
        "tensors": [
            {"data": X, "role": "streamed"},              # X — bank-resident, streams through PE
            {"data": Y, "role": "accumulator"},            # Y — partial-sum / result
        ],
        "scalars": [A_SCALAR, N],
    }


# ── Standalone CPU verification ──────────────────────────────────────

def _verify() -> int:
    """Quick sanity check: compile via IM, run on CPU, compare with NumPy."""
    import ctypes
    from triton.backends.im import IMTarget, compile_im_kernel, launch_im_kernel
    from triton.compiler import ASTSource, make_backend

    cfg = pim_kernel_config()
    num_programs = cfg["grid"][0]

    # Compile
    src = ASTSource(cfg["kernel"], cfg["signature"],
                    constexprs=cfg["constants"], attrs={})
    target = IMTarget("hbm-pim", NUM_BANKS)
    backend = make_backend(target)
    opts = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=target, options=opts.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")
    lib = compile_im_kernel(llir, kernel_name="axpy_kernel")

    # Prepare data
    X_np = cfg["tensors"][0]["data"]
    Y_np = cfg["tensors"][1]["data"].copy()
    Y_ref = A_SCALAR * X_np + cfg["tensors"][1]["data"]

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    X_c = X_np.ctypes.data_as(INT32_P)
    Y_c = Y_np.ctypes.data_as(INT32_P)

    launch_im_kernel(lib, kernel_name="axpy_kernel",
                     num_banks=NUM_BANKS, num_programs=num_programs,
                     arg_types=[INT32_P, INT32_P, ctypes.c_int32,
                                ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p],
                     arg_values=[X_c, Y_c, A_SCALAR, N, None, None])

    if np.array_equal(Y_np, Y_ref):
        print(f"[PASS] All {N} elements match")
        return 0
    mismatches = int(np.sum(Y_np != Y_ref))
    print(f"[FAIL] {mismatches} element(s) differ")
    return 1


if __name__ == "__main__":
    sys.exit(_verify())
