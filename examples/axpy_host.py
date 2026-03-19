"""
axpy_host.py — AXPY (Y = a*X + Y) in Triton, running on the host CPU.

Uses TRITON_INTERPRET=1 (pure-Python tensor interpreter) so no GPU is
required.  The kernel still goes through the full Triton front-end and
optimization pipeline (TTIR → TTGIR lowering passes) before the
interpreter executes it, which lets you inspect the generated IR.

Usage
-----
    # Activate the venv that has triton installed:
    source third_party/triton/.venv/bin/activate

    # Run with the interpreter (no GPU needed):
    TRITON_INTERPRET=1 python examples/axpy_host.py

    # Print the TTIR (Triton IR) before optimization:
    TRITON_INTERPRET=1 MLIR_ENABLE_DUMP=1 python examples/axpy_host.py

    # Validate only (no run, just show expected vs actual):
    TRITON_INTERPRET=1 python examples/axpy_host.py --validate
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Auto-enable interpreter mode only when the env-var is not set at all.
# If the user explicitly sets TRITON_INTERPRET=0, respect that choice.
# ---------------------------------------------------------------------------
if "TRITON_INTERPRET" not in os.environ:
    print(
        "[axpy_host] TRITON_INTERPRET is not set.\n"
        "Re-launching with TRITON_INTERPRET=1 so the kernel runs on CPU …\n"
    )
    os.environ["TRITON_INTERPRET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def axpy_kernel(
    X_ptr,          # pointer to input  X  (float32)
    Y_ptr,          # pointer to in/out Y  (float32)
    a,              # scalar multiplier
    N,              # number of elements
    BLOCK: tl.constexpr,
):
    """Y[i] = a * X[i] + Y[i]  for i in [0, N)."""
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < N
    x     = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y     = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Y_ptr + offs, a * x + y, mask=mask)


# ---------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------

def axpy(a: float, x: torch.Tensor, y: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Run AXPY through Triton and return the updated y tensor."""
    assert x.shape == y.shape and x.ndim == 1
    n = x.numel()

    grid = (triton.cdiv(n, block),)
    axpy_kernel[grid](x, y, a, n, BLOCK=block)
    return y


def main():
    parser = argparse.ArgumentParser(description="Triton AXPY on host CPU via interpreter")
    parser.add_argument("--n",     type=int,   default=1024,  help="Vector length")
    parser.add_argument("--a",     type=float, default=2.0,   help="Scalar multiplier")
    parser.add_argument("--block", type=int,   default=64,    help="Triton BLOCK size (constexpr)")
    parser.add_argument("--seed",  type=int,   default=42,    help="Random seed")
    parser.add_argument("--validate", action="store_true",    help="Only validate, suppress full output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.rand(args.n, dtype=torch.float32)
    y = torch.rand(args.n, dtype=torch.float32)
    ref = args.a * x + y.clone()

    result = axpy(args.a, x, y.clone(), block=args.block)

    max_err = torch.max(torch.abs(result - ref)).item()
    passed  = max_err < 1e-5

    print(f"N={args.n}  a={args.a}  BLOCK={args.block}")
    print(f"Max absolute error vs numpy: {max_err:.3e}")
    print(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if not args.validate:
        np.set_printoptions(precision=4, suppress=True)
        preview = min(8, args.n)
        print(f"\nFirst {preview} elements of result:  {result[:preview].cpu().numpy()}")
        print(f"First {preview} elements of torch ref: {ref[:preview].cpu().numpy()}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
