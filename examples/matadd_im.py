"""
matadd_im.py — Compile a 2D matrix-add kernel via the Triton IM backend.

Demonstrates a 2D grid:  program_id(0) tiles rows, program_id(1) tiles cols.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
import triton.language as tl
from triton.backends.im import IMTarget
from triton.compiler import ASTSource, make_backend

NUM_WARPS = 1
NUM_CTAS = 1


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

    a = tl.load(A + offs, mask, other=0) + 5
    b = tl.load(B + offs, mask, other=0)
    tl.store(C + offs, a + b, mask)


def build_compile_context(debug: bool = False):
    signature = {
        "A": "*i32",
        "B": "*i32",
        "C": "*i32",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
    }
    constants = {"BLOCK_M": 4, "BLOCK_N": 64}
    attrs = {}

    src = ASTSource(matadd_kernel, signature, constexprs=constants, attrs=attrs)
    target = IMTarget("hbm-pim", 16, debug=debug)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS, "num_ctas": NUM_CTAS})
    return src, target, backend, options


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton IM MatAdd to LLVM IR")
    parser.add_argument("--out", type=str, default=None, help="Write full LLVM IR to this path")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Dump IR after intermediate compiler passes")
    args = parser.parse_args()

    src, target, backend, options = build_compile_context(debug=args.debug)
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    ll = ccinfo.asm[backend.binary_ext]
    if isinstance(ll, (bytes, bytearray)):
        ll = ll.decode("utf-8")

    if args.out:
        Path(args.out).write_text(ll, encoding="utf-8")
        print(f"[matadd_im] wrote {len(ll)} bytes to {args.out}")


if __name__ == "__main__":
    main()
