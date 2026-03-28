#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_triton_im_ramulator2_e2e.sh
#
# Thin wrapper: activates the Triton venv and delegates to the Python
# e2e script.
#
# Usage:
#   ./scripts/run_triton_im_ramulator2_e2e.sh examples/axpy_pim_e2e.py [extra flags...]
#   ./scripts/run_triton_im_ramulator2_e2e.sh examples/matadd_pim_e2e.py --skip-ramulator
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/run_triton_im_ramulator2_e2e.py"
TRACER_DIR="$ROOT_DIR/third_party/ramulator2/llvm-tracer"

# Activate the Triton venv
for venv in "$ROOT_DIR/third_party/triton/.venv/bin/activate" \
            "$ROOT_DIR/third_party/triton/.venv314/bin/activate"; do
  if [[ -f "$venv" ]]; then
    source "$venv"
    break
  fi
done

# Build MemTracePass.dylib if missing
if [[ ! -f "$TRACER_DIR/build/MemTracePass.dylib" ]]; then
  echo "[pre] Building llvm-tracer (MemTracePass.dylib) …"
  make -C "$TRACER_DIR"
fi

cd "$ROOT_DIR"
exec python "$SCRIPT" "$@"
