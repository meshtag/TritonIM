"""
matvec_im.py — Compile a matrix-vector multiply kernel via the Triton IM backend.

Demonstrates a K-reduction loop: y = A @ x  where A is [M, K], x is [K].
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
def matvec_kernel(A, x, y, M, K, BLOCK: tl.constexpr):
    """y[i] = sum_k A[i, k] * x[k], tiled over BLOCK output rows."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in range(K):
        a_val = tl.load(A + offs * K + k, mask, other=0)
        x_val = tl.load(x + k)
        acc += a_val * x_val * 5
    tl.store(y + offs, acc, mask)


def build_compile_context(debug: bool = False):
    signature = {
        "A": "*i32",
        "x": "*i32",
        "y": "*i32",
        "M": "i32",
        "K": "i32",
        "BLOCK": "constexpr",
    }
    constants = {"BLOCK": 64}
    # tt.divisibility=16 for pointers (always aligned) and size args
    # (assumed 16-aligned) — enables AxisInfo vectorisation.
    d16 = [["tt.divisibility", 16]]
    attrs = {(0,): d16, (1,): d16, (2,): d16, (3,): d16, (4,): d16}

    src = ASTSource(matvec_kernel, signature, constexprs=constants, attrs=attrs)
    target = IMTarget("hbm-pim", 32, debug=debug)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS, "num_ctas": NUM_CTAS})
    return src, target, backend, options


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton IM MatVec to LLVM IR")
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
        print(f"[matvec_im] wrote {len(ll)} bytes to {args.out}")


if __name__ == "__main__":
    main()
