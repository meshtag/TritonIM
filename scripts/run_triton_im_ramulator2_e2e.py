#!/usr/bin/env python3
"""
run_triton_im_ramulator2_e2e.py — Triton IM kernel → PIM trace → Ramulator2

Full pipeline:
  1. Load user's Python file (must define ``pim_kernel_config()``)
  2. Compile the Triton kernel through the IM backend → LLVM IR
  3. Strip IR for CPU, instrument with MemTracePass
  4. Compile instrumented IR + im_runtime.c + pim_runtime.c → shared lib
  5. Execute kernel across all (pid, bank) pairs with PIM tracing active
  6. Run Ramulator2 HBM3_PIM simulation on the generated trace

Usage::

    python scripts/run_triton_im_ramulator2_e2e.py examples/axpy_pim_e2e.py
    python scripts/run_triton_im_ramulator2_e2e.py examples/matadd_pim_e2e.py --skip-ramulator

The input Python file must define ``pim_kernel_config()`` returning a dict::

    {
        'kernel':       <@triton.jit function>,
        'kernel_name':  str,
        'signature':    dict,          # ASTSource signature
        'constants':    dict,          # constexprs
        'num_banks':    int,           # default 32
        'grid':         tuple,         # (nx,) or (nx, ny) or (nx, ny, nz)
        'tensors': [                   # one per pointer arg, in order
            {'data': np.ndarray, 'role': 'streamed'|'operand'|'accumulator'},
            ...
        ],
        'scalars':      list,          # scalar arg values, in order after tensors
    }
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton  # noqa: E402
from triton.backends.im import IMTarget  # noqa: E402
from triton.backends.im.launcher import strip_for_cpu  # noqa: E402
from triton.compiler import ASTSource, make_backend  # noqa: E402

# ── Repo-relative paths ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
TRACER_DIR = ROOT / "third_party" / "ramulator2" / "llvm-tracer"
IM_RUNTIME_C = TRACER_DIR / "runtime" / "im_runtime.c"
PIM_RUNTIME_C = TRACER_DIR / "runtime" / "pim_runtime.c"
SIMDRAM_RUNTIME_C = TRACER_DIR / "runtime" / "simdram_runtime.c"
MEM_TRACE_PASS = TRACER_DIR / "build" / "MemTracePass.dylib"
COMPUTE_TRACE_PASS = TRACER_DIR / "build" / "ComputeTracePass.dylib"
HBMPIM_CONFIG = TRACER_DIR / "config" / "hbmpim_config.yaml"
SIMDRAM_CONFIG = TRACER_DIR / "config" / "simdram_hbm3.yaml"
RAMULATOR2_BIN = ROOT / "third_party" / "ramulator2" / "build" / "ramulator2"

# MemTracePass.dylib is built against system LLVM 18; use matching opt.
OPT_BIN = os.environ.get("OPT", "/usr/local/bin/opt")

# PIM role enum (must match pim_runtime.h)
_PIM_ROLE = {"streamed": 0, "operand": 1, "accumulator": 2}
_PIM_PHASE_IDLE = 0
_PIM_PHASE_COMPUTE = 1

# Map Triton type-strings → ctypes scalars
_SCALAR_CTYPE = {
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}


# ── Helpers ───────────────────────────────────────────────────────────


def _validate_bank_count(cfg: dict, ramulator_config: str | None) -> None:
    """Warn if num_banks in the kernel config disagrees with the Ramulator2 YAML.

    In Samsung HBM-PIM, there is one PE per bank.  The Ramulator2 model
    parameterises this as ``pe_per_bankgroup`` (= banks_per_bankgroup for
    the 1:1 real design).  The Triton IM backend's ``num_banks`` must
    equal the total PE count (= num_pch × num_bg × pe_per_bankgroup).
    """
    config_path = ramulator_config or str(HBMPIM_CONFIG)
    if not os.path.isfile(config_path):
        return
    with open(config_path) as f:
        text = f.read()

    # Parse HBM3_PIM preset to get default pch/bg
    presets = {
        "HBM3_2Gb": (2, 4),  # (pch, bg)
        "HBM3_4Gb": (2, 4),
        "HBM3_8Gb": (2, 4),
    }
    preset_match = re.search(r"preset:\s*(HBM3_\w+)", text)
    if not preset_match or preset_match.group(1) not in presets:
        return
    n_pch, n_bg = presets[preset_match.group(1)]

    # pe_per_bankgroup (default 1 if not specified)
    m = re.search(r"pe_per_bankgroup:\s*(\d+)", text)
    pe_per_bg = int(m.group(1)) if m else 1

    num_pes = n_pch * n_bg * pe_per_bg

    user_banks = cfg.get("num_banks", 32)
    if user_banks != num_pes:
        print(
            f"[WARN] Bank/PE count mismatch: kernel config has num_banks={user_banks} "
            f"but Ramulator2 config ({config_path}) implies "
            f"{n_pch} pch × {n_bg} bg × {pe_per_bg} PE/bg = {num_pes} PEs.\n"
            f"       num_banks should equal the number of PEs for accurate PIM simulation.",
            file=sys.stderr,
        )


def _load_user_config(filepath: str) -> dict:
    """Import the user's Python file and call pim_kernel_config()."""
    spec = importlib.util.spec_from_file_location("_user_kernel", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "pim_kernel_config"):
        print(
            f"[ERROR] {filepath} must define pim_kernel_config()",
            file=sys.stderr,
        )
        sys.exit(1)
    return mod.pim_kernel_config()


def _compile_to_llir(cfg: dict) -> str:
    """Triton kernel → IM backend → LLVM IR string."""
    src = ASTSource(
        cfg["kernel"],
        cfg["signature"],
        constexprs=cfg["constants"],
        attrs=cfg.get("attrs", {}),
    )
    num_banks = cfg.get("num_banks", 32)
    target = IMTarget("hbm-pim", num_banks)
    backend = make_backend(target)
    opts = backend.parse_options({"num_warps": 1, "num_ctas": 1})
    ccinfo = triton.compile(src, target=target, options=opts.__dict__)
    llir = ccinfo.asm[backend.binary_ext]
    if isinstance(llir, (bytes, bytearray)):
        llir = llir.decode("utf-8")
    return llir


def _instrument_ir(llir: str, work_dir: str, target: str = "hbm-pim") -> str:
    """Strip for CPU → opt + MemTracePass (+ ComputeTracePass for SIMDRAM)."""
    clean = strip_for_cpu(llir)
    clean_path = os.path.join(work_dir, "kernel_clean.ll")
    with open(clean_path, "w") as f:
        f.write(clean)

    inst_path = os.path.join(work_dir, "kernel_instrumented.ll")

    if target == "simdram":
        # Apply both MemTracePass (loads/stores) and ComputeTracePass (arith)
        subprocess.check_call(
            [
                OPT_BIN,
                f"--load-pass-plugin={MEM_TRACE_PASS}",
                f"--load-pass-plugin={COMPUTE_TRACE_PASS}",
                "--passes=mem-trace,compute-trace",
                "-S",
                clean_path,
                "-o",
                inst_path,
            ]
        )
    else:
        subprocess.check_call(
            [
                OPT_BIN,
                f"--load-pass-plugin={MEM_TRACE_PASS}",
                "--passes=mem-trace",
                "-S",
                clean_path,
                "-o",
                inst_path,
            ]
        )
    return inst_path


def _compile_shared_lib(inst_ir_path: str, work_dir: str, target: str = "hbm-pim") -> str:
    """Compile instrumented IR + runtimes into a shared lib."""
    ext = "dylib" if platform.system() == "Darwin" else "so"
    lib_name = "libkernel_simdram" if target == "simdram" else "libkernel_pim"
    lib_path = os.path.join(work_dir, f"{lib_name}.{ext}")
    runtime_c = str(SIMDRAM_RUNTIME_C) if target == "simdram" else str(PIM_RUNTIME_C)
    subprocess.check_call(
        [
            "clang",
            "-shared",
            "-O2",
            "-fPIC",
            "-Wno-override-module",
            inst_ir_path,
            str(IM_RUNTIME_C),
            runtime_c,
            "-o",
            lib_path,
        ]
    )
    return lib_path


def _scalar_ctype(sig_dict: dict, name: str) -> type:
    """Resolve a scalar arg's ctypes type from the Triton signature."""
    ty_str = sig_dict.get(name, "i32")
    return _SCALAR_CTYPE.get(ty_str, ctypes.c_int32)


def _run_pim_trace(cfg: dict, lib_path: str, trace_file: str, target: str = "hbm-pim") -> None:
    """Load shared lib, drive PIM/SIMDRAM-traced execution, produce trace file."""
    lib = ctypes.CDLL(lib_path)

    # ── Runtime init/register/phase/finalize ──
    if target == "simdram":
        init_fn = lib.simdram_init
        register_fn = lib.simdram_register_tensor
        phase_fn = lib.simdram_set_phase
        finalize_fn = lib.simdram_finalize
    else:
        init_fn = lib.pim_init
        register_fn = lib.pim_register_tensor
        phase_fn = lib.pim_set_phase
        finalize_fn = lib.pim_finalize

    init_fn.argtypes = [ctypes.c_char_p]
    init_fn.restype = None
    init_fn(trace_file.encode())

    register_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    register_fn.restype = ctypes.c_int

    phase_fn.argtypes = [ctypes.c_int]
    phase_fn.restype = None

    finalize_fn.argtypes = []
    finalize_fn.restype = None

    # ── IM runtime (bank / program-id state) ──
    for fn_name in (
        "__pim_set_bank_id",
        "__pim_set_program_id",
        "__pim_set_program_id_y",
        "__pim_set_program_id_z",
    ):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.c_int32]
        fn.restype = None

    set_bank = lib.__pim_set_bank_id
    set_pid = lib.__pim_set_program_id
    set_pid_y = lib.__pim_set_program_id_y
    set_pid_z = lib.__pim_set_program_id_z

    # ── Register tensors with PIM runtime ──
    kept_refs = []  # prevent GC of numpy arrays
    tensor_ptrs = []

    for t in cfg["tensors"]:
        arr = np.ascontiguousarray(t["data"])
        role = _PIM_ROLE[t["role"]]
        shape = arr.shape
        dims_c = (ctypes.c_int * len(shape))(*shape)
        ptr = arr.ctypes.data_as(ctypes.c_void_p)

        tid = register_fn(ptr, dims_c, len(shape), arr.dtype.itemsize, role)
        if tid < 0:
            sys.exit(f"[ERROR] register_tensor failed (shape={shape}, role={t['role']})")

        kept_refs.append(arr)
        tensor_ptrs.append(ptr)

    # ── Assemble kernel args ──
    sig = cfg["signature"]
    sig_names = [k for k, v in sig.items() if v != "constexpr"]
    ptr_names = [k for k, v in sig.items() if isinstance(v, str) and v.startswith("*")]
    scalar_names = [k for k in sig_names if k not in ptr_names]

    arg_types = []
    arg_values = []

    for ptr_val in tensor_ptrs:
        arg_types.append(ctypes.c_void_p)
        arg_values.append(ptr_val)

    scalars = cfg.get("scalars", [])
    for i, sval in enumerate(scalars):
        sname = scalar_names[i] if i < len(scalar_names) else None
        cty = _scalar_ctype(sig, sname) if sname else ctypes.c_int32
        arg_types.append(cty)
        arg_values.append(sval)

    # Triton always appends two scratch-pointer args
    arg_types += [ctypes.c_void_p, ctypes.c_void_p]
    arg_values += [None, None]

    kernel_fn = getattr(lib, cfg["kernel_name"])
    kernel_fn.argtypes = arg_types
    kernel_fn.restype = None

    # ── Execute ──
    grid = cfg["grid"]
    nx = grid[0]
    ny = grid[1] if len(grid) > 1 else 1
    nz = grid[2] if len(grid) > 2 else 1
    num_banks = cfg.get("num_banks", 32)

    total = nx * ny * nz * num_banks
    print(f"[exec] grid=({nx},{ny},{nz})  banks={num_banks}  invocations={total}")

    phase_fn(_PIM_PHASE_COMPUTE)

    for pz in range(nz):
        set_pid_z(pz)
        for py in range(ny):
            set_pid_y(py)
            for px in range(nx):
                set_pid(px)
                for bank in range(num_banks):
                    set_bank(bank)
                    kernel_fn(*arg_values)

    phase_fn(_PIM_PHASE_IDLE)
    finalize_fn()


def _run_ramulator2(trace_file: str, config_path: str | None = None, target: str = "hbm-pim") -> int:
    """Run Ramulator2 HBM3_PIM on the trace. Returns exit code."""
    if not RAMULATOR2_BIN.is_file():
        print(f"[WARN] Ramulator2 not found: {RAMULATOR2_BIN}")
        return -1

    default_cfg = str(SIMDRAM_CONFIG) if target == "simdram" else str(HBMPIM_CONFIG)
    base_cfg = config_path or default_cfg
    with open(base_cfg) as f:
        cfg_text = f.read()
    cfg_text = re.sub(r"path:.*", f"path: {trace_file}", cfg_text)

    tmp_cfg = trace_file.replace(".txt", "_config.yaml")
    with open(tmp_cfg, "w") as f:
        f.write(cfg_text)

    print(f"[ram]  config={tmp_cfg}")
    r = subprocess.run(
        [str(RAMULATOR2_BIN), "--config_file", tmp_cfg],
        capture_output=True,
        text=True,
    )
    if r.stdout:
        print(r.stdout)
    if r.stderr:
        # Ramulator2 prints stats to stderr
        for line in r.stderr.splitlines():
            print(f"       {line}")

    os.remove(tmp_cfg)
    return r.returncode


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E2E: Triton IM kernel → PIM trace → Ramulator2",
    )
    parser.add_argument("kernel_file", help="Python file defining pim_kernel_config()")
    parser.add_argument("--target", default="hbm-pim", choices=["hbm-pim", "simdram"],
                        help="PIM target: hbm-pim (default) or simdram")
    parser.add_argument("--trace", default=None, help="Output trace path (default: <workdir>/<target>_trace.txt)")
    parser.add_argument("--work-dir", default=None, help="Dir for intermediate files")
    parser.add_argument("--skip-ramulator", action="store_true", help="Skip Ramulator2")
    parser.add_argument("--ramulator-config", default=None, help="Override Ramulator2 config YAML")
    args = parser.parse_args()

    # Validate prerequisites
    missing = []
    prereqs = [
        (IM_RUNTIME_C, "im_runtime.c"),
        (MEM_TRACE_PASS, "MemTracePass.dylib (run 'make' in llvm-tracer/)"),
    ]
    if args.target == "simdram":
        prereqs += [
            (SIMDRAM_RUNTIME_C, "simdram_runtime.c"),
            (COMPUTE_TRACE_PASS, "ComputeTracePass.dylib (run 'make' in llvm-tracer/)"),
            (SIMDRAM_CONFIG, "simdram_hbm3.yaml"),
        ]
    else:
        prereqs += [
            (PIM_RUNTIME_C, "pim_runtime.c"),
            (HBMPIM_CONFIG, "hbmpim_config.yaml"),
        ]
    for p, label in prereqs:
        if not p.is_file():
            missing.append(f"  {label}: {p}")
    if missing:
        print("[ERROR] Missing prerequisites:\n" + "\n".join(missing), file=sys.stderr)
        return 1

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="triton_pim_e2e_")
    default_trace = "simdram_trace.txt" if args.target == "simdram" else "pim_trace.txt"
    trace_file = args.trace or os.path.join(work_dir, default_trace)
    os.makedirs(work_dir, exist_ok=True)

    print(f"[info] work_dir   = {work_dir}")
    print(f"[info] trace_file = {trace_file}")

    # Step 1 — Load user config
    print(f"\n[1/6] Loading kernel config from {args.kernel_file}")
    cfg = _load_user_config(args.kernel_file)
    print(
        f"       kernel={cfg['kernel_name']}  grid={cfg['grid']}  "
        f"banks={cfg.get('num_banks', 32)}  "
        f"tensors={len(cfg['tensors'])}  scalars={len(cfg.get('scalars', []))}"
    )

    if args.target != "simdram":
        _validate_bank_count(cfg, args.ramulator_config)

    # Step 2 — Triton → LLVM IR
    print("\n[2/6] Compiling Triton kernel via IM backend")
    llir = _compile_to_llir(cfg)
    raw_path = os.path.join(work_dir, "kernel_raw.ll")
    with open(raw_path, "w") as f:
        f.write(llir)
    print(f"       → {len(llir)} bytes  ({raw_path})")

    # Step 3 — Instrument IR
    if args.target == "simdram":
        print("\n[3/6] Instrumenting loads/stores + arithmetic (MemTracePass + ComputeTracePass)")
    else:
        print("\n[3/6] Instrumenting loads/stores (MemTracePass)")
    inst_path = _instrument_ir(llir, work_dir, target=args.target)
    print(f"       → {inst_path}")

    # Step 4 — Compile shared library
    rt_label = "simdram_runtime" if args.target == "simdram" else "pim_runtime"
    print(f"\n[4/6] Building shared library (kernel + im_runtime + {rt_label})")
    lib_path = _compile_shared_lib(inst_path, work_dir, target=args.target)
    print(f"       → {lib_path}")

    # Step 5 — Execute with tracing
    target_label = "SIMDRAM" if args.target == "simdram" else "PIM"
    print(f"\n[5/6] Executing kernel with {target_label} tracing → {trace_file}")
    _run_pim_trace(cfg, lib_path, trace_file, target=args.target)

    with open(trace_file) as f:
        n_ops = sum(1 for _ in f)
    print(f"       → {n_ops} PIM trace operations")

    # Step 6 — Ramulator2
    if args.skip_ramulator:
        print("\n[6/6] Ramulator2 skipped (--skip-ramulator)")
    else:
        sim_label = "SIMDRAM" if args.target == "simdram" else "HBM3_PIM"
        print(f"\n[6/6] Running Ramulator2 {sim_label} simulation")
        rc = _run_ramulator2(trace_file, args.ramulator_config, target=args.target)
        if rc not in (0, -1):
            print(f"[WARN] Ramulator2 exited with code {rc}")

    print(f"\n{'='*60}")
    print(f"[DONE] Trace: {trace_file}  ({n_ops} ops)")
    print(f"       Work:  {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
