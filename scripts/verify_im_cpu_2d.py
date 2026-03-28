#!/usr/bin/env python3
"""
verify_im_cpu_2d.py — Functional verification of the Triton IM backend (2D kernel).

Workflow
-------
  1. Compile the 2D matrix-add kernel via the IM backend → LLVM IR
  2. Compile IR + im_runtime.c into a CPU-native shared library
  3. Execute the kernel across all ``(program_id_y, program_id_x, bank_id)`` pairs
  4. Compare C element-by-element against a NumPy reference

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
def matadd_kernel(A, B, C, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """C[i, j] = A[i, j] + B[i, j], tiled in 2D."""
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offs = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2D index: rows[:, None] * N + cols[None, :]
    offs = row_offs[:, None] * N + col_offs[None, :]
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)

    a = tl.load(A + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    tl.store(C + offs, a + b, mask)


# ── Configuration ─────────────────────────────────────────────────────

BLOCK_M   = 4           # rows per program invocation
BLOCK_N   = 64          # columns per program invocation
NUM_BANKS = 16          # HBM-PIM banks
M         = 16          # matrix rows
N         = 256         # matrix columns


# ── Helpers ───────────────────────────────────────────────────────────

def compile_matadd_ir() -> str:
    """Compile MatAdd through the IM backend, return LLVM IR string."""
    src = ASTSource(
        matadd_kernel,
        {"A": "*i32", "B": "*i32", "C": "*i32", "M": "i32", "N": "i32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
        constexprs={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
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
    num_programs_x = math.ceil(M / BLOCK_M)   # row tiles
    num_programs_y = math.ceil(N / BLOCK_N)    # column tiles
    print(f"[config] M={M}  N={N}  BLOCK_M={BLOCK_M}  BLOCK_N={BLOCK_N}  "
          f"NUM_BANKS={NUM_BANKS}")
    print(f"         grid=({num_programs_x}, {num_programs_y})  "
          f"sizePerThread={BLOCK_N // NUM_BANKS}")

    # 1. Triton → LLVM IR
    print("[step 1] Compiling MatAdd kernel via IM backend …")
    llir = compile_matadd_ir()
    print(f"         → {len(llir)} bytes of LLVM IR")

    # 2. IR + im_runtime.c → shared library
    print("[step 2] Building CPU-native shared library …")
    lib = compile_im_kernel(llir, kernel_name="matadd_kernel")

    # 3. Prepare data  (row-major flat arrays)
    A_np = np.arange(1, M * N + 1, dtype=np.int32).reshape(M, N)
    B_np = np.arange(10, 10 * (M * N) + 1, 10, dtype=np.int32).reshape(M, N)
    C_ref = A_np + B_np
    C_np = np.zeros((M, N), dtype=np.int32)

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_flat = A_np.ravel()
    B_flat = B_np.ravel()
    C_flat = C_np.ravel()

    A_c = A_flat.ctypes.data_as(INT32_P)
    B_c = B_flat.ctypes.data_as(INT32_P)
    C_c = C_flat.ctypes.data_as(INT32_P)

    # 4. Execute across all (program_id_y, program_id_x, bank_id) pairs
    print("[step 3] Executing kernel (2D grid, sequential bank emulation) …")
    launch_im_kernel(
        lib,
        kernel_name="matadd_kernel",
        num_banks=NUM_BANKS,
        num_programs=num_programs_x,
        num_programs_y=num_programs_y,
        arg_types=[INT32_P, INT32_P, INT32_P,
                   ctypes.c_int32, ctypes.c_int32,
                   ctypes.c_void_p, ctypes.c_void_p],
        arg_values=[A_c, B_c, C_c, M, N, None, None],
    )

    # 5. Compare
    print("[step 4] Comparing results …")
    C_result = np.ctypeslib.as_array(C_c, shape=(M * N,)).reshape(M, N)
    mismatches = 0

    print(C_result)
    print("\n")
    print(C_ref)

    for i in range(M):
        for j in range(N):
            if C_result[i, j] != C_ref[i, j]:
                print(f"  MISMATCH [{i},{j}]: got {C_result[i, j]}, "
                      f"expected {C_ref[i, j]}")
                mismatches += 1
                if mismatches > 10:
                    print("  … (output truncated)")
                    break
        if mismatches > 10:
            break

    if mismatches == 0:
        print(f"\n[PASS] All {M}×{N} = {M * N} elements match the NumPy "
              f"reference (C = A + B)")
        return 0
    else:
        print(f"\n[FAIL] {mismatches} element(s) differ out of {M * N}")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
