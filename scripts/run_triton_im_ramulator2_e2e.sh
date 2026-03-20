#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRITON_VENV="$ROOT_DIR/third_party/triton/.venv/bin/activate"
TRITON_DEBUG="$ROOT_DIR/test_im_debug.py"
TRACER_DIR="$ROOT_DIR/third_party/ramulator2/llvm-tracer"
RAMULATOR_BIN="$ROOT_DIR/third_party/ramulator2/build/ramulator2"
TRACER_SCRIPT="$TRACER_DIR/run_pim_trace.sh"
HBMPIM_CONFIG="$TRACER_DIR/config/hbmpim_config.yaml"
DEFAULT_INPUT_C="$TRACER_DIR/test/axpy_pim.c"

if [[ ! -f "$TRITON_VENV" ]]; then
  echo "[ERROR] Triton venv not found: $TRITON_VENV"
  exit 1
fi

if [[ ! -f "$TRITON_DEBUG" ]]; then
  echo "[ERROR] Triton debug script not found: $TRITON_DEBUG"
  exit 1
fi

if [[ ! -x "$TRACER_SCRIPT" ]]; then
  echo "[ERROR] Tracer script not executable/missing: $TRACER_SCRIPT"
  exit 1
fi

INPUT_C="${1:-$DEFAULT_INPUT_C}"
PIM_TRACE_OUT="${PIM_TRACE_OUT:-pim_trace.txt}"

if [[ ! -f "$INPUT_C" ]]; then
  echo "[ERROR] Input C file not found: $INPUT_C"
  echo "Usage: $0 [path/to/input.c]"
  exit 1
fi

echo "[1/3] Triton IM compile sanity check"
(
  cd "$ROOT_DIR"
  source "$TRITON_VENV"
  TRITON_BACKENDS_IN_TREE=1 python "$TRITON_DEBUG"
)

echo "[2/3] Build llvm-tracer artifacts (if needed)"
(
  cd "$TRACER_DIR"
  make
)

echo "[3/3] Generate PIM trace and run Ramulator2"
(
  cd "$TRACER_DIR"
  source "$TRITON_VENV"
  RAMULATOR2="$RAMULATOR_BIN" \
  PIM_TRACE_OUT="$PIM_TRACE_OUT" \
  "$TRACER_SCRIPT" "$INPUT_C" "$HBMPIM_CONFIG"
)

echo ""
echo "[DONE] E2E run complete"
echo "- Trace file: $TRACER_DIR/$PIM_TRACE_OUT"
echo "- Triton check: passed via test_im_debug.py"
echo "- Ramulator config: $HBMPIM_CONFIG"
