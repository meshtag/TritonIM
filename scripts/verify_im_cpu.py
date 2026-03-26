#!/usr/bin/env python3
"""
verify_im_cpu.py — Functional verification of the Triton IM backend.

Workflow
-------
  1. Compile the AXPY kernel via the IM backend → LLVM IR
  2. Compile IR + im_runtime.c into a CPU-native shared library
     (via :func:`compile_im_kernel`)
  3. Execute the kernel across all ``(program_id, bank_id)`` pairs
     (via :func:`launch_im_kernel`)
  4. Compare Y element-by-element against a NumPy reference

Exit code: 0 on success, 1 on mismatch.
"""

import ctypes
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Triton imports
# ---------------------------------------------------------------------------
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton                                                    # noqa: E402
import triton.language as tl                                     # noqa: E402
from triton.backends.im import (                                 # noqa: E402
    IMTarget, compile_im_kernel, launch_im_kernel,
)
from triton.compiler import ASTSource, make_backend              # noqa: E402


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

BLOCK     = 64          # elements per program invocation
NUM_BANKS = 16          # HBM-PIM banks (4 BG × 4 banks)
N         = 512         # total elements  (= 4 programs × 64)
A_SCALAR  = 3           # AXPY constant


# ── Helpers ───────────────────────────────────────────────────────────

def compile_axpy_ir() -> str:
    """Compile AXPY through the IM backend, return LLVM IR string."""
    src = ASTSource(
        axpy_kernel,
        {"X": "*i32", "Y": "*i32", "A": "i32", "N": "i32",
         "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK},
        attrs={},
    )
    target = IMTarget("hbm-pim", NUM_BANKS)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")
    return llir


# ── Main ──────────────────────────────────────────────────────────────

def verify() -> int:
    num_programs = math.ceil(N / BLOCK)
    print(f"[config] N={N}  BLOCK={BLOCK}  NUM_BANKS={NUM_BANKS}  "
          f"A={A_SCALAR}  num_programs={num_programs}  "
          f"sizePerThread={BLOCK // NUM_BANKS}")

    # 1. Triton → LLVM IR
    print("[step 1] Compiling AXPY kernel via IM backend …")
    llir = compile_axpy_ir()
    print(f"         → {len(llir)} bytes of LLVM IR")

    # 2. IR + im_runtime.c → shared library (compile_im_kernel handles
    #    post-processing: strips addrspace, target triple, etc.)
    print("[step 2] Building CPU-native shared library …")
    lib = compile_im_kernel(llir, kernel_name="axpy_kernel")

    # 3. Prepare data
    X_np = np.arange(1, N + 1, dtype=np.int32)              # [1 .. N]
    Y_np = np.arange(10, 10 * N + 1, 10, dtype=np.int32)    # [10, 20 .. 10N]
    Y_ref = A_SCALAR * X_np + Y_np                           # NumPy reference

    Y_work = Y_np.copy()
    INT32_P = ctypes.POINTER(ctypes.c_int32)
    X_c = X_np.ctypes.data_as(INT32_P)
    Y_c = Y_work.ctypes.data_as(INT32_P)

    # 4. Execute across all (program_id, bank_id) pairs
    print("[step 3] Executing kernel (sequential bank emulation) …")
    launch_im_kernel(
        lib,
        kernel_name="axpy_kernel",
        num_banks=NUM_BANKS,
        num_programs=num_programs,
        arg_types=[INT32_P, INT32_P, ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_void_p, ctypes.c_void_p],
        arg_values=[X_c, Y_c, A_SCALAR, N, None, None],
    )

    # print(Y_work)
    # print("\n")
    # print(Y_ref)

    # 5. Compare
    print("[step 4] Comparing results …")
    mismatches = 0
    for i in range(N):
        if Y_work[i] != Y_ref[i]:
            print(f"  MISMATCH [{i:3d}]: got {Y_work[i]}, expected {Y_ref[i]}")
            mismatches += 1
            if mismatches > 10:
                print("  … (output truncated)")
                break

    if mismatches == 0:
        print(f"\n[PASS] All {N} elements match the NumPy reference "
              f"(Y = {A_SCALAR}·X + Y)")
        return 0
    else:
        print(f"\n[FAIL] {mismatches} element(s) differ out of {N}")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
