#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# t4_full_suite.sh — Ouroboros v7.1 Wave 12 full verification battery (T4 box)
#
# Runs on a Colab T4 host with the four Ouroboros repos cloned side-by-side
# under one PARENT directory (ouroboros-triton, ouroboros-core, ouroboros-dfg,
# ECC) plus torch+CUDA+triton and pytest installed.
#
# Usage:
#   bash scripts/t4_full_suite.sh [PARENT]
#
#     PARENT  directory containing the four repos. Defaults to this script's
#             grandparent (<repo>/scripts/../..), i.e. the side-by-side clone
#             root.
#
# Sections:
#   A  environment report      nvidia-smi, torch/CUDA/triton versions
#   B  full triton suite       python3 -m pytest tests/ -q   (gates exit code)
#   C  permutation-aware study d_head=8 channel-permutation bridge proof
#                              (reference_attention.perm/apply_channel_perm +
#                              a random dense computation; no kernel needed)
#   D  placement-head sweep    scaling_study.run_config/render_table over
#                              {d_model=16,ep=6,inst=128},{32,8,256},{64,10,384}
#                              (gates exit code: any non-finite lift fails)
#
# Exit status is non-zero ONLY if B fails or D reports any non-finite lift.
# All output is teed to stdout and a timestamped log next to this script.
# ---------------------------------------------------------------------------

set -u

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PARENT="${1:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

TRITON_REPO="$PARENT/ouroboros-triton"
CORE_ROOT="$PARENT/ouroboros-core"
BATTERY="$TRITON_REPO/scripts/wave12_t4_battery.py"
export CORE_ROOT

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp"
LOG_FILE="$LOG_DIR/t4_full_suite_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE" 2>/dev/null || LOG_FILE="/tmp/t4_full_suite_$(date +%Y%m%d_%H%M%S).log"

# Tee everything the suite emits to stdout AND the log file.
exec > >(tee "$LOG_FILE") 2>&1

B_RC=0
C_RC=0
D_RC=0

banner() {
    echo ""
    echo "======================================================================="
    echo "== $*"
    echo "======================================================================="
}

echo "t4_full_suite.sh — started $(date -Is)"
echo "PARENT     : $PARENT"
echo "triton repo: $TRITON_REPO"
echo "core repo  : $CORE_ROOT"
echo "log file   : $LOG_FILE"

if [ ! -d "$TRITON_REPO/tests" ]; then
    echo "FATAL: $TRITON_REPO/tests not found — pass the correct PARENT as arg1."
    exit 2
fi
if [ ! -f "$BATTERY" ]; then
    echo "FATAL: battery module missing: $BATTERY"
    exit 2
fi

# ---------------------------------------------------------------------------
# Section A — environment report
# ---------------------------------------------------------------------------
banner "SECTION A: environment report"

echo "--- nvidia-smi ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "(nvidia-smi returned non-zero — GPU may be unattached)"
else
    echo "(nvidia-smi not on PATH — CPU-only host?)"
fi

echo ""
echo "--- python / torch / CUDA / triton versions ---"
python3 - <<'PYEOF'
import importlib.util
import platform
import sys

print("python      :", sys.version.split()[0], f"({platform.platform()})")
for mod in ("torch", "triton"):
    spec = importlib.util.find_spec(mod)
    if spec is None:
        print(f"{mod:<12}: NOT INSTALLED")
        continue
    m = __import__(mod)
    ver = getattr(m, "__version__", "?")
    extra = ""
    if mod == "torch":
        extra = f"  cuda_available={torch_cuda}" if (torch_cuda := m.cuda.is_available()) else \
                f"  cuda_available=False (build: {getattr(m.version, 'cuda', 'none')})"
        if m.cuda.is_available():
            extra += f"  device={m.cuda.get_device_name(0)}"
    print(f"{mod:<12}: {ver}{extra}")
PYEOF

# ---------------------------------------------------------------------------
# Section B — full triton test suite (GPU-gated legs skip or run per host)
# ---------------------------------------------------------------------------
banner "SECTION B: full triton suite — python3 -m pytest tests/ -q"

cd "$TRITON_REPO" || { echo "FATAL: cannot cd $TRITON_REPO"; exit 2; }
python3 -m pytest tests/ -q
B_RC=$?
echo "Section B pytest exit code: $B_RC"

# ---------------------------------------------------------------------------
# Section C — permutation-aware d_head=8 study (no kernel required)
# ---------------------------------------------------------------------------
banner "SECTION C: permutation-aware d_head=8 study (perm/apply_channel_perm bridge)"

cd "$TRITON_REPO" || { echo "FATAL: cannot cd $TRITON_REPO"; exit 2; }
python3 - <<'PYEOF'
import json
import sys

sys.path.insert(0, "scripts")          # wave12_t4_battery lives here
import wave12_t4_battery as battery

result = battery.perm_study(d_head=8)
print(json.dumps(result, indent=2))
unperm = float(result["unpermuted_max_diff"])
perm = float(result["permuted_max_diff"])
print(f"[perm-study] un-permuted max_diff = {unperm:.6f}")
print(f"[perm-study]   permuted  max_diff = {perm:.3e}")
assert perm < 1e-6 < unperm, (
    f"permutation bridge failed: permuted={perm:.3e} must be << "
    f"un-permuted={unperm:.6f}"
)
print("[perm-study] PASS: permuted max_diff << un-permuted max_diff")
PYEOF
C_RC=$?
if [ "$C_RC" -ne 0 ]; then
    echo "!!! Section C FAILED (rc=$C_RC) — permutation bridge study did not hold."
else
    echo "Section C exit code: $C_RC (OK)"
fi

# ---------------------------------------------------------------------------
# Section D — placement-head sweep through ouroboros-core scaling_study
# ---------------------------------------------------------------------------
banner "SECTION D: placement-head sweep — scaling_study configs {16,6ep,128} {32,8ep,256} {64,10ep,384}"

cd "$TRITON_REPO" || { echo "FATAL: cannot cd $TRITON_REPO"; exit 2; }
python3 - <<'PYEOF'
import math
import os
import sys

sys.path.insert(0, "scripts")  # battery module

import wave12_t4_battery as battery
if os.environ.get("CORE_ROOT"):
    battery.set_core_src(os.environ["CORE_ROOT"])

rows = battery.head_sweep()

print()
print("[head-sweep] rendered scaling table (scaling_study.render_table):")
print(battery.render_table(rows))
print()
print(f"[head-sweep] device: {rows[0].get('_device')}  "
      f"per-cell seconds: {[r.get('_seconds') for r in rows]}")

bad = [r for r in rows if not math.isfinite(float(r["lift"]))]
for r in rows:
    print(
        f"[head-sweep] d_model={r['d_model']} epochs={r['epochs']} "
        f"instances={r['instances']}: lift={r['lift']:.6f} "
        f"finite={math.isfinite(float(r['lift']))}"
    )
if bad:
    print(f"FATAL: {len(bad)} sweep row(s) reported a NON-FINITE lift: "
          f"{[(r.get('d_model'), r.get('lift')) for r in bad]}")
    sys.exit(1)
print("[head-sweep] PASS: all lifts finite")
PYEOF
D_RC=$?
if [ "$D_RC" -ne 0 ]; then
    echo "!!! Section D FAILED (rc=$D_RC) — non-finite lift or scaling_study error."
else
    echo "Section D exit code: $D_RC (OK)"
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
banner "VERDICT"
echo "A: environment report   (informational)"
echo "B: triton pytest rc     : $B_RC   <- gates exit code"
echo "C: perm-study rc        : $C_RC   (report-only)"
echo "D: head-sweep rc        : $D_RC   <- gates exit code (non-finite lift)"

if [ "$B_RC" -ne 0 ] || [ "$D_RC" -ne 0 ]; then
    echo "OVERALL: FAIL (log: $LOG_FILE)"
    exit 1
fi
echo "OVERALL: PASS (log: $LOG_FILE)"
exit 0
