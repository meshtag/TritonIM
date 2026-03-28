"""Debug script: step through IM backend pipeline stages."""
import os
os.environ["TRITON_BACKENDS_IN_TREE"] = "1"

import triton
import triton.language as tl
from triton.backends.im import IMTarget
from triton.compiler import ASTSource, make_backend

@triton.jit
def axpy_kernel(X, Y, A, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask, other=0)
    y = tl.load(Y + offs, mask, other=0)
    out = A * x + y
    tl.store(Y + offs, out, mask)

src = ASTSource(
    axpy_kernel,
    {"X": "*i32", "Y": "*i32", "A": "i32", "N": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": 64},
    attrs={},
)
target = IMTarget("hbm-pim", 32)
backend = make_backend(target)
options = backend.parse_options({"num_warps": 1, "num_ctas": 1})

ccinfo = triton.compile(src, target=target, options=options.__dict__)
ll = ccinfo.asm[backend.binary_ext]
if isinstance(ll, (bytes, bytearray)):
    ll = ll.decode("utf-8")
print("=== LLVM IR (first 40 lines) ===")
for line in ll.splitlines()[:40]:
    print(line)
print(f"\n[OK] {len(ll)} bytes of LLVM IR generated")
