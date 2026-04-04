#!/usr/bin/env python3
"""
matmul_2d_int_e2e.py — 2D-tiled integer matrix multiply for PIM backends.

============================================================================
OVERVIEW
============================================================================

Implements  C = A @ B   where  A is [M, K],  B is [K, N],  C is [M, N],
all in int32.  Supports both **HBM-PIM** and **SIMDRAM** targets via the
``--target`` flag.  The Triton kernel is identical for both; only tensor
roles and bank count differ.

============================================================================
TARGET DIFFERENCES
============================================================================

  Aspect          HBM-PIM                      SIMDRAM
  ─────────────   ──────────────────────────   ──────────────────────────
  A role          STREAMED (stream thru PE)     STREAMED (write to bit-rows)
  B role          OPERAND  (load to PE reg)     STREAMED (write to bit-rows)
  C role          ACCUMULATOR (R/BW)            ACCUMULATOR (R + MAJ BR)
  Compute         PE executes MAC               MAJ-3 gates in subarray
  Float support   Yes                           No (integer only)
  NUM_BANKS       32                            1  (trace-size limited)

In SIMDRAM both operands must reside in subarray rows for MAJ computation,
so B is STREAMED (not OPERAND).

============================================================================
COST MODEL (SIMDRAM — from paper Table 1)
============================================================================

  Operation       Cost (n = pe_bits)
  ─────────────   ───────────────────
  multiplication  11n² − 5n − 1
  addition        8n + 1
  subtraction     8n + 1
  division        8n² + 12n
  bitwise AND     n
  bitwise OR      n
  bitwise XOR     5n

For n = 16:  mul = 2735,  add = 129.  Each maps to one BR trace op.

============================================================================
USAGE
============================================================================

E2E pipeline (compile → trace → Ramulator2)::

    # SIMDRAM target
    python scripts/run_triton_im_ramulator2_e2e.py \\
        examples/matmul_2d_int_e2e.py --target simdram

    # HBM-PIM target
    python scripts/run_triton_im_ramulator2_e2e.py \\
        examples/matmul_2d_int_e2e.py --target hbm-pim

Standalone CPU verification::

    python examples/matmul_2d_int_e2e.py --target simdram
    python examples/matmul_2d_int_e2e.py --target hbm-pim
    python examples/matmul_2d_int_e2e.py --target simdram --debug
"""

import math
import os
import sys

import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
# KERNEL  (identical for both HBM-PIM and SIMDRAM — same Triton code)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def matmul_2d_int_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """C[i,j] = sum_k A[i,k] * B[k,j], with 2D output tiling."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row_offs < M
    col_mask = col_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(K):
        a_ptrs = A + row_offs * K + k
        a_col  = tl.load(a_ptrs, mask=row_mask, other=0)

        b_ptrs = B + k * N + col_offs
        b_row  = tl.load(b_ptrs, mask=col_mask, other=0)

        acc += a_col[:, None] * b_row[None, :]

    c_ptrs = row_offs[:, None] * N + col_offs[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(C + c_ptrs, acc, mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BLOCK_M = 4
BLOCK_N = 4

# Matrix dimensions (integer).
# Kept small because SIMDRAM trace explodes: every integer multiply
# emits 2735 BR ops (11n²−5n−1 at n=16) and every add emits 129 BR ops.
M = 16
N = 16
K = 8

GRID_M = math.ceil(M / BLOCK_M)
GRID_N = math.ceil(N / BLOCK_N)

# Per-target defaults
TARGET_CFG = {
    "simdram": {
        # NUM_BANKS=1: trace-size limited.  In a full SIMDRAM config
        # (16ch × 2pch × 4bg × 4banks/bg) there are 512 physical banks,
        # but each bank runs the full kernel independently, making the
        # trace grow as NUM_BANKS × grid × K × BLOCK_M × BLOCK_N ×
        # (mul_cost + add_cost).
        "num_banks": 1,
        "b_role":    "streamed",   # both operands must be in subarray rows
    },
    "hbm-pim": {
        "num_banks": 32,
        "b_role":    "operand",    # B loaded to PE register via data bus
    },
}

# ── Resolve which target to use ──────────────────────────────────
# When loaded by the e2e script, PIM_TARGET is set from --target.
# When run standalone, argparse sets it (see __main__).
PIM_TARGET = os.environ.get("PIM_TARGET", "simdram")


# ═══════════════════════════════════════════════════════════════════════
# E2E CONFIG (required by run_triton_im_ramulator2_e2e.py)
# ═══════════════════════════════════════════════════════════════════════

def pim_kernel_config():
    """Return the kernel + data configuration for the e2e pipeline.

    Tensor roles differ by target:
      HBM-PIM:  A=STREAMED  B=OPERAND   C=ACCUMULATOR
      SIMDRAM:  A=STREAMED  B=STREAMED  C=ACCUMULATOR
    """
    tcfg = TARGET_CFG[PIM_TARGET]
    num_banks = tcfg["num_banks"]
    b_role = tcfg["b_role"]

    A = np.arange(1, M * K + 1, dtype=np.int32).reshape(M, K)
    B = np.arange(1, K * N + 1, dtype=np.int32).reshape(K, N)
    C = np.zeros((M, N), dtype=np.int32)

    sig = {
        "A": "*i32", "B": "*i32", "C": "*i32",
        "M": "i32", "N": "i32", "K": "i32",
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
    }

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
        "kernel":      matmul_2d_int_kernel,
        "kernel_name": "matmul_2d_int_kernel",
        "signature":   sig,
        "constants":   {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        "attrs":       attrs,
        "num_banks":   num_banks,
        "grid":        (GRID_M, GRID_N),
        "tensors": [
            {"data": A.ravel(), "role": "streamed"},      # A — always streamed
            {"data": B.ravel(), "role": b_role},           # B — target-dependent
            {"data": C.ravel(), "role": "accumulator"},    # C — result
        ],
        "scalars": [M, N, K],
    }


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE CPU VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def _verify(target: str = "simdram", debug: bool = False) -> int:
    """Compile, execute on CPU, compare against NumPy's A @ B."""
    import ctypes
    from triton.backends.im import IMTarget, compile_im_kernel, launch_im_kernel
    from triton.compiler import ASTSource, make_backend

    tcfg = TARGET_CFG[target]
    num_banks = tcfg["num_banks"]

    cfg = pim_kernel_config()
    grid_m, grid_n = cfg["grid"]

    src = ASTSource(
        cfg["kernel"],
        cfg["signature"],
        constexprs=cfg["constants"],
        attrs=cfg["attrs"],
    )
    im_target = IMTarget(target, num_banks, debug=debug)
    backend = make_backend(im_target)
    opts = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=im_target, options=opts.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")

    lib = compile_im_kernel(llir, kernel_name="matmul_2d_int_kernel")

    A_np = cfg["tensors"][0]["data"].reshape(M, K)
    B_np = cfg["tensors"][1]["data"].reshape(K, N)
    C_np = cfg["tensors"][2]["data"].copy()

    C_ref = (A_np @ B_np).ravel()

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_c = cfg["tensors"][0]["data"].ctypes.data_as(INT32_P)
    B_c = cfg["tensors"][1]["data"].ctypes.data_as(INT32_P)
    C_c = C_np.ctypes.data_as(INT32_P)

    launch_im_kernel(
        lib,
        kernel_name="matmul_2d_int_kernel",
        num_banks=num_banks,
        num_programs=grid_m,
        num_programs_y=grid_n,
        arg_types=[
            INT32_P, INT32_P, INT32_P,
            ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p,
        ],
        arg_values=[A_c, B_c, C_c, M, N, K, None, None],
    )

    if np.array_equal(C_np, C_ref):
        print(f"[PASS] matmul_2d_int ({target}): all {M}×{N} = {M * N} "
              f"output elements match (K={K}, tile={BLOCK_M}×{BLOCK_N}, "
              f"grid={GRID_M}×{GRID_N}, banks={num_banks})")
        return 0

    mismatches = int(np.sum(C_np != C_ref))
    diff_idx = np.where(C_np != C_ref)[0][:10]
    for idx in diff_idx:
        row, col = divmod(int(idx), N)
        print(f"  C[{row},{col}] = {C_np[idx]}  expected {C_ref[idx]}")
    print(f"[FAIL] matmul_2d_int ({target}): {mismatches}/{M * N} "
          f"element(s) differ")
    return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="2D-tiled integer matmul for PIM (HBM-PIM / SIMDRAM)")
    parser.add_argument("--target", default="simdram",
                        choices=["hbm-pim", "simdram"],
                        help="PIM target (default: simdram)")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Dump IR after intermediate compiler passes")
    args = parser.parse_args()
    PIM_TARGET = args.target
    sys.exit(_verify(target=args.target, debug=args.debug))
