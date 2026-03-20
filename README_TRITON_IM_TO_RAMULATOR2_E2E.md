# Triton IM -> LLVM IR -> Ramulator2 Trace: End-to-End Verification

This README captures the current verified flow in this workspace and provides
copy-paste commands to validate each stage.

## What this verifies

1. **Triton IM backend lowering works**: Triton kernel compiles to LLVM IR.
2. **LLVM tracer pipeline works**: LLVM IR/C workload gets instrumented and emits PIM traces.
3. **Ramulator2 HBM-PIM simulation works**: generated trace is replayed by Ramulator2.

## One-command runner

Use the helper script to run the two-stage verification automatically:

```bash
cd /Users/meshtag/TritonPIM
./scripts/run_triton_im_ramulator2_e2e.sh
```

Optional custom tracer input C file:

```bash
cd /Users/meshtag/TritonPIM
./scripts/run_triton_im_ramulator2_e2e.sh \
    /Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer/test/attention_pim.c
```

The script performs:
- Triton IM compile sanity (`test_im_debug.py`)
- `llvm-tracer` build (`make`)
- Trace generation + Ramulator2 run (`run_pim_trace.sh`)

> Note: In the current repo state, stages (1) and (2)/(3) are both verified.
> A single automated bridge script that directly consumes Triton-emitted kernel LLVM
> IR and runs it through the tracer pipeline is not present at repo root.

---

## Paths used below

- Repo root: `/Users/meshtag/TritonPIM`
- Triton test driver: `test_im_debug.py`
- Triton virtualenv: `third_party/triton/.venv`
- Triton tools: `third_party/triton/build/cmake.macosx-12.1-arm64-cpython-3.13/bin`
- LLVM tracer: `third_party/ramulator2/llvm-tracer`
- MemTrace pass: `third_party/ramulator2/llvm-tracer/build/MemTracePass.dylib`
- Ramulator2 binary: `third_party/ramulator2/build/ramulator2`
- HBM-PIM config: `third_party/ramulator2/llvm-tracer/config/hbmpim_config.yaml`

---

## 0) One-time build sanity (if needed)

```bash
cd /Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer
make
```

This should produce:
- `build/MemTracePass.dylib`
- `build/libtrace_runtime.dylib`

And ensure Ramulator2 exists:

```bash
ls /Users/meshtag/TritonPIM/third_party/ramulator2/build/ramulator2
```

---

## 1) Verify Triton IM -> LLVM IR

Run the existing debug driver:

```bash
cd /Users/meshtag/TritonPIM
source third_party/triton/.venv/bin/activate
TRITON_BACKENDS_IN_TREE=1 python test_im_debug.py
```

Expected result:
- Prints first ~40 lines of LLVM IR
- Ends with `[OK] <N> bytes of LLVM IR generated`

### (Optional) Save full Triton-generated LLVM IR to a file

```bash
cd /Users/meshtag/TritonPIM
source third_party/triton/.venv/bin/activate
mkdir -p artifacts
TRITON_BACKENDS_IN_TREE=1 python - <<'PY'
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
    constexprs={"BLOCK": 16},
    attrs={},
)
target = IMTarget("im", "pim", 1)
backend = make_backend(target)
options = backend.parse_options({"num_warps": 1, "num_ctas": 1})
ccinfo = triton.compile(src, target=target, options=options.__dict__)
ll = ccinfo.asm[backend.binary_ext]
if isinstance(ll, (bytes, bytearray)):
    ll = ll.decode("utf-8")
out = "artifacts/triton_axpy_kernel.ll"
with open(out, "w") as f:
    f.write(ll)
print(f"wrote {out} ({len(ll)} bytes)")
PY
```

---

## 2) Verify LLVM tracing + PIM trace generation + Ramulator2 simulation

Use the tracer's built-in end-to-end script with a known PIM test input:

```bash
cd /Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer
source /Users/meshtag/TritonPIM/third_party/triton/.venv/bin/activate
RAMULATOR2=/Users/meshtag/TritonPIM/third_party/ramulator2/build/ramulator2 \
PIM_TRACE_OUT=pim_trace.txt \
./run_pim_trace.sh test/axpy_pim.c config/hbmpim_config.yaml
```

This runs:
1. C -> LLVM IR
2. `opt -O2`
3. MemTrace instrumentation
4. executable run to emit `pim_trace.txt`
5. Ramulator2 replay using HBM3_PIM config

Expected artifacts in `third_party/ramulator2/llvm-tracer/`:
- `test/axpy_pim.ll`
- `test/axpy_pim.opt.ll`
- `test/axpy_pim.instrumented.ll`
- `test/axpy_pim.pim.exe`
- `pim_trace.txt`

---

## 3) Quick checks for outputs

```bash
cd /Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer
wc -l pim_trace.txt
head -n 10 pim_trace.txt
```

(Optional) capture Ramulator stats log:

```bash
cd /Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer
RAMULATOR2=/Users/meshtag/TritonPIM/third_party/ramulator2/build/ramulator2 \
./run_pim_trace.sh test/axpy_pim.c config/hbmpim_config.yaml | tee ramulator_run.log
```

---

## Current gap to a single-command Triton->Trace flow

What is already working:
- Triton IM backend emits LLVM IR for Triton kernels.
- LLVM-tracer + Ramulator2 produces valid PIM traces and HBM-PIM simulation results.

What remains for a fully direct one-command flow:
- A small bridge utility that takes Triton-produced LLVM IR + a host harness, then invokes
  MemTrace instrumentation and Ramulator2 automatically.

Until that direct bridge script is added, use this README's commands (or the helper script)
as the full verification path.

---

## Triton-side design notes

Detailed Triton implementation notes for the IM backend changes are in:

- `third_party/triton/README_IM_BACKEND_CHANGES.md`
