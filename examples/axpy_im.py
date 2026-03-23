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
def axpy_kernel(X, Y, A, N, BLOCK: tl.constexpr):
    # One program handles one block of elements.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask, other=0)
    y = tl.load(Y + offs, mask, other=0)
    out = A * x + y
    tl.store(Y + offs, out, mask)

def build_compile_context():
    signature = {
        "X": "*i32",
        "Y": "*i32",
        "A": "i32",
        "N": "i32",
        "BLOCK": "constexpr",
    }
    # BLOCK=64 with 16 banks → sizePerThread=4
    #   Level 1 (SIMT-like):  16 banks process different data in parallel
    #   Level 2 (SIMD-like):  each bank handles 4 elements sequentially / vectorized
    constants = {"BLOCK": 64}
    attrs = {}

    src = ASTSource(axpy_kernel, signature, constexprs=constants, attrs=attrs)
    target = IMTarget("hbm-pim", 16, debug=True)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS, "num_ctas": NUM_CTAS})
    return src, target, backend, options


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton IM AXPY to LLVM IR")
    parser.add_argument("--out", type=str, default=None, help="Write full LLVM IR to this path")
    parser.add_argument("--preview", action="store_true", help="Print only the first 30 lines")
    parser.add_argument("--print", action="store_true", help="Print full LLVM IR to stdout")
    args = parser.parse_args()

    src, target, backend, options = build_compile_context()
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    ll = ccinfo.asm[backend.binary_ext]
    if isinstance(ll, (bytes, bytearray)):
        ll = ll.decode("utf-8")

    if args.out:
        Path(args.out).write_text(ll, encoding="utf-8")
        print(f"[axpy_im] wrote {len(ll)} bytes to {args.out}")

    if args.preview:
        print("\n".join(ll.splitlines()[:30]))
    elif args.print or not args.out:
        print(ll)


if __name__ == "__main__":
    main()
