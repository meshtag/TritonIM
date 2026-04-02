#!/usr/bin/env python3
"""
matmul_2d_pim_e2e.py — 2D-tiled matrix multiply for the Triton-IM backend.

============================================================================
OVERVIEW
============================================================================

Implements  C = A @ B   where  A is [M, K],  B is [K, N],  C is [M, N].

Unlike the 1D matmul example (matmul_pim_e2e.py), this kernel uses
**true 2D tiling** — each program instance owns a rectangular
BLOCK_M × BLOCK_N tile of the output matrix C, and the reduction
over K proceeds one element at a time within that 2D tile.

============================================================================
KERNEL DESIGN — 2D TILED MATMUL
============================================================================

Grid layout (2D):

    program_id(0) = pid_m → indexes BLOCK_M-sized row tiles of C and A
    program_id(1) = pid_n → indexes BLOCK_N-sized column tiles of C and B

    Total grid = ceil(M / BLOCK_M) × ceil(N / BLOCK_N) program instances.

Each program instance:

  1. Computes row offsets:  row_offs = pid_m * BLOCK_M + arange(0, BLOCK_M)
     Shape: (BLOCK_M,)

  2. Computes col offsets:  col_offs = pid_n * BLOCK_N + arange(0, BLOCK_N)
     Shape: (BLOCK_N,)

  3. Allocates a 2D accumulator:
         acc = zeros((BLOCK_M, BLOCK_N), dtype=int32)

  4. Iterates over K (step = 1):

     for k in range(K):
         a_col = load(A + row_offs[:, None] * K + k)
             → loads A[row, k] for each row in the tile
             → shape: (BLOCK_M, 1)  (broadcast along columns)

         b_row = load(B + k * N + col_offs[None, :])
             → loads B[k, col] for each col in the tile
             → shape: (1, BLOCK_N)  (broadcast along rows)

         acc += a_col * b_row
             → outer product: (BLOCK_M, 1) × (1, BLOCK_N) → (BLOCK_M, BLOCK_N)
             → each element acc[i, j] accumulates A[row_i, k] * B[k, col_j]

  5. Stores the 2D tile:
         store(C + row_offs[:, None] * N + col_offs[None, :], acc)

============================================================================
COMPARISON: 1D vs 2D TILING
============================================================================

                    1D (matmul_pim_e2e.py)         2D (this file)
  ─────────────────────────────────────────────────────────────────
  Grid              1D (flat over M*N)              2D (pid_m × pid_n)
  Block shape       (BLOCK,)                        (BLOCK_M, BLOCK_N)
  Accumulator       1D vector                       2D tile
  A load per k      A[row, k] per element           A[row, k] per row (broadcast)
  B load per k      B[k, col] per element           B[k, col] per col (broadcast)
  Data reuse        None — each element loads       A column reused across BLOCK_N
                    independently                   B row reused across BLOCK_M
  Output indexing   flat → (row, col) via div/mod   direct 2D: row_offs, col_offs

The 2D version is structurally closer to the canonical Triton matmul and
exposes the data-reuse pattern that makes tiled matmul efficient. On a GPU,
this reuse matters for shared memory and L1 cache. On a PIM accelerator,
the reuse means fewer host→PE data transfers per output element.

============================================================================
PIM MAPPING
============================================================================

The 2D grid maps program instances to memory bank groups. Within each
program instance, banks process BLOCK_M × BLOCK_N / num_banks elements
of the output tile.

Per K-step, the PIM operations are:
  - A (STREAMED  → BR): bank-resident weights, each PE reads its rows
  - B (OPERAND   → W):  input row B[k, :], loaded to PE registers
  - C (ACCUMULATOR → BW): result tile, written once after K loop

============================================================================
USAGE
============================================================================

E2E pipeline (compile → trace → Ramulator2)::

    python scripts/run_triton_im_ramulator2_e2e.py examples/matmul_2d_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/matmul_2d_pim_e2e.py --skip-ramulator

Standalone CPU verification::

    python examples/matmul_2d_pim_e2e.py
"""

import math
import os
import sys

import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════
# KERNEL
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def matmul_2d_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """C[i,j] = sum_k A[i,k] * B[k,j], with 2D output tiling.

    Each program instance owns a BLOCK_M × BLOCK_N tile of C.
    The K-reduction is a scalar loop that accumulates outer products
    of A-column and B-row slices into the 2D accumulator.
    """
    # ── Tile coordinates ──────────────────────────────────────────
    pid_m = tl.program_id(0)       # which BLOCK_M-row tile
    pid_n = tl.program_id(1)       # which BLOCK_N-col tile

    # Row and column offsets for this tile
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    # Bounds masks
    row_mask = row_offs < M        # (BLOCK_M,)
    col_mask = col_offs < N        # (BLOCK_N,)

    # ── 2D accumulator ───────────────────────────────────────────
    # Shape: (BLOCK_M, BLOCK_N) — one element per (row, col) in the tile.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # ── K-reduction loop ─────────────────────────────────────────
    for k in range(K):
        # Load A[:, k] for this row tile → shape (BLOCK_M, 1)
        # A is row-major: A[row, k] = A + row * K + k
        a_ptrs = A + row_offs * K + k               # (BLOCK_M,) pointers
        a_col  = tl.load(a_ptrs, mask=row_mask, other=0)  # (BLOCK_M,)

        # Load B[k, :] for this col tile → shape (1, BLOCK_N)
        # B is row-major: B[k, col] = B + k * N + col
        b_ptrs = B + k * N + col_offs               # (BLOCK_N,) pointers
        b_row  = tl.load(b_ptrs, mask=col_mask, other=0)  # (BLOCK_N,)

        # Outer product: (BLOCK_M, 1) × (1, BLOCK_N) → (BLOCK_M, BLOCK_N)
        # a_col[:, None] broadcasts a_col along the column dimension
        # b_row[None, :] broadcasts b_row along the row dimension
        acc += a_col[:, None] * b_row[None, :]

    # ── Store 2D result tile ─────────────────────────────────────
    # C is row-major: C[row, col] = C + row * N + col
    c_ptrs = row_offs[:, None] * N + col_offs[None, :]   # (BLOCK_M, BLOCK_N)
    c_mask = row_mask[:, None] & col_mask[None, :]        # (BLOCK_M, BLOCK_N)
    tl.store(C + c_ptrs, acc, mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Tile dimensions — each program instance handles a BLOCK_M × BLOCK_N
# rectangular sub-tile of the output matrix C.
BLOCK_M   = 4     # rows per tile    (must divide M for simplicity)
BLOCK_N   = 64    # columns per tile (must divide N for simplicity)

NUM_BANKS = 32    # PIM banks (= threads_per_warp in IM backend)

# Matrix dimensions
M         = 512    # rows of A / rows of C
N         = 512    # cols of B / cols of C
K         = 32    # reduction dimension (cols of A / rows of B)

# Derived grid dimensions
GRID_M    = math.ceil(M / BLOCK_M)   # = 128  program instances along rows
GRID_N    = math.ceil(N / BLOCK_N)   # = 8  program instances along cols


# ═══════════════════════════════════════════════════════════════════════
# PIM E2E CONFIG (required by run_triton_im_ramulator2_e2e.py)
# ═══════════════════════════════════════════════════════════════════════

def pim_kernel_config():
    """Return the kernel + data configuration for the e2e pipeline.

    This function is called by run_triton_im_ramulator2_e2e.py to get:
    - The Triton kernel function and its signature
    - Input/output numpy arrays with test data
    - Grid dimensions, bank count, and tensor roles for PIM trace generation
    """
    # ── Test data ─────────────────────────────────────────────────
    # Use small sequential integers so results are easy to verify by hand.
    # A: [M, K] = [16, 32], values 1..512
    # B: [K, N] = [32, 64], values 1..2048
    # C: [M, N] = [16, 64], initially zeros → filled with A @ B
    A = np.arange(1, M * K + 1, dtype=np.int32).reshape(M, K)
    B = np.arange(1, K * N + 1, dtype=np.int32).reshape(K, N)
    C = np.zeros((M, N), dtype=np.int32)

    # ── Kernel signature ──────────────────────────────────────────
    sig = {
        "A": "*i32", "B": "*i32", "C": "*i32",
        "M": "i32", "N": "i32", "K": "i32",
        "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
    }

    # ── Divisibility attributes ───────────────────────────────────
    # tt.divisibility=16 for pointer args (always aligned) and scalar
    # args whose values are divisible by 16. This enables AxisInfo-based
    # vectorization in the IM backend, producing vector loads/stores.
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
        "kernel":      matmul_2d_kernel,
        "kernel_name": "matmul_2d_kernel",
        "signature":   sig,
        "constants":   {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        "attrs":       attrs,
        "num_banks":   NUM_BANKS,
        # 2D grid: (GRID_M, GRID_N) = (4, 1)
        "grid":        (GRID_M, GRID_N),
        "tensors": [
            {"data": A.ravel(), "role": "streamed"},      # A — bank-resident
            {"data": B.ravel(), "role": "operand"},       # B — loaded to PE
            {"data": C.ravel(), "role": "accumulator"},   # C — result
        ],
        "scalars": [M, N, K],
    }


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE CPU VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def _verify() -> int:
    """Compile matmul_2d_kernel via the IM backend, execute on CPU,
    and compare every output element against NumPy's A @ B.

    Returns 0 on success, 1 on mismatch.
    """
    import ctypes
    from triton.backends.im import IMTarget, compile_im_kernel, launch_im_kernel
    from triton.compiler import ASTSource, make_backend

    cfg = pim_kernel_config()
    grid_m, grid_n = cfg["grid"]

    # ── Step 1: Compile ───────────────────────────────────────────
    # Build a Triton ASTSource from the kernel + signature, then compile
    # through the IM backend pipeline:
    #   Triton AST → Triton IR → TritonIM dialect → LLVM IR → shared lib
    src = ASTSource(
        cfg["kernel"],
        cfg["signature"],
        constexprs=cfg["constants"],
        attrs=cfg["attrs"],
    )
    target  = IMTarget("hbm-pim", NUM_BANKS)
    backend = make_backend(target)
    opts    = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo  = triton.compile(src, target=target, options=opts.__dict__)
    llir    = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")

    # compile_im_kernel: LLVM IR → .so (clang JIT on macOS / Linux)
    lib = compile_im_kernel(llir, kernel_name="matmul_2d_kernel")

    # ── Step 2: Prepare data ──────────────────────────────────────
    A_np = cfg["tensors"][0]["data"].reshape(M, K)
    B_np = cfg["tensors"][1]["data"].reshape(K, N)
    C_np = cfg["tensors"][2]["data"].copy()   # flat (M*N,), all zeros

    # Reference: NumPy integer matmul
    C_ref = (A_np @ B_np).ravel()

    INT32_P = ctypes.POINTER(ctypes.c_int32)
    A_c = cfg["tensors"][0]["data"].ctypes.data_as(INT32_P)
    B_c = cfg["tensors"][1]["data"].ctypes.data_as(INT32_P)
    C_c = C_np.ctypes.data_as(INT32_P)

    # ── Step 3: Execute ───────────────────────────────────────────
    # launch_im_kernel iterates over:
    #   for pid_y in range(grid_n):       ← program_id(1) = column tile
    #     for pid in range(grid_m):       ← program_id(0) = row tile
    #       for bank in range(num_banks): ← bank_id within each tile
    #         kernel(args...)
    launch_im_kernel(
        lib,
        kernel_name="matmul_2d_kernel",
        num_banks=NUM_BANKS,
        num_programs=grid_m,         # axis 0: row tiles
        num_programs_y=grid_n,       # axis 1: column tiles
        arg_types=[
            INT32_P, INT32_P, INT32_P,          # A, B, C pointers
            ctypes.c_int32, ctypes.c_int32,     # M, N
            ctypes.c_int32,                     # K
            ctypes.c_void_p, ctypes.c_void_p,   # global_scratch, profile_scratch
        ],
        arg_values=[A_c, B_c, C_c, M, N, K, None, None],
    )

    # ── Step 4: Verify ────────────────────────────────────────────
    if np.array_equal(C_np, C_ref):
        print(f"[PASS] matmul_2d: all {M}×{N} = {M * N} output elements "
              f"match (K={K}, tile={BLOCK_M}×{BLOCK_N}, "
              f"grid={GRID_M}×{GRID_N})")
        return 0

    mismatches = int(np.sum(C_np != C_ref))
    # Print first few mismatches for debugging
    diff_idx = np.where(C_np != C_ref)[0][:10]
    for idx in diff_idx:
        row, col = divmod(int(idx), N)
        print(f"  C[{row},{col}] = {C_np[idx]}  expected {C_ref[idx]}")
    print(f"[FAIL] matmul_2d: {mismatches}/{M * N} element(s) differ")
    return 1


if __name__ == "__main__":
    sys.exit(_verify())
