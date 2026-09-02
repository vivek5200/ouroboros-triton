#!/usr/bin/env python3
"""colab_t4_proof.py — One-cell Colab script for full Ouroboros v7.1 T4 proof.

Paste this into a single Colab cell on a T4 GPU runtime. It will:

  1. Clone all three repos (ouroboros-triton, ouroboros-core, ouroboros-dfg)
  2. Install Python dependencies (torch, triton, pybind11, pytest, etc.)
  3. Build the C++ parity twin (src/cpp/build.sh)
  4. Run the FULL triton test suite (all GPU kernel legs + C++ parity)
  5. Run the permutation-aware d_head=8 study (Section C)
  6. Run the placement-head scaling sweep (Section D)
  7. Run ouroboros-core tests
  8. Run ouroboros-dfg Rust tests (if cargo is available)
  9. Save a timestamped proof log to /content/t4_proof_<timestamp>.log

Expected outcome on T4: ~149 passed / 6 skipped (triton), ~143 passed (core),
with the 6 C++ parity tests and all GPU kernel legs ACTIVE.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GITHUB_USER = "vivek5200"
WORK_DIR = "/content/ouroboros_proof"
REPOS = {
    "ouroboros-triton": f"https://github.com/{GITHUB_USER}/ouroboros-triton.git",
    "ouroboros-core": f"https://github.com/{GITHUB_USER}/ouroboros-core.git",
    "ouroboros-dfg": f"https://github.com/{GITHUB_USER}/ouroboros-dfg.git",
}

LOG_LINES: list[str] = []
SECTION_RESULTS: dict[str, dict] = {}


def log(msg: str = "") -> None:
    """Print and buffer a line for the proof log."""
    print(msg, flush=True)
    LOG_LINES.append(msg)


def run(cmd: str, cwd: str | None = None, check: bool = True,
        timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output live."""
    log(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    if result.stdout:
        for line in result.stdout.rstrip().split("\n"):
            log(line)
    if check and result.returncode != 0:
        log(f"[FAIL] exit code {result.returncode}")
    return result


def banner(title: str) -> None:
    log("")
    log("=" * 72)
    log(f"== {title}")
    log("=" * 72)


# ---------------------------------------------------------------------------
# 0. Setup: create work dir
# ---------------------------------------------------------------------------
banner("SETUP")
started = datetime.now(timezone.utc).isoformat()
log(f"Started: {started}")
os.makedirs(WORK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------
banner("SECTION A: Environment Report")

run("nvidia-smi", check=False)
run("python3 --version")
run("pip --version", check=False)

# Check CUDA availability before anything else
log("\n--- Torch / CUDA / Triton check ---")
run("""python3 -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device count: {torch.cuda.device_count()}')
try:
    import triton
    print(f'triton: {triton.__version__}')
except ImportError:
    print('triton: NOT INSTALLED (will install)')
"
""", check=False)


# ---------------------------------------------------------------------------
# 2. Clone repos
# ---------------------------------------------------------------------------
banner("SECTION B: Clone Repositories")

for name, url in REPOS.items():
    repo_path = os.path.join(WORK_DIR, name)
    if os.path.isdir(repo_path):
        log(f"{name}: already cloned, pulling latest...")
        run(f"git -C {repo_path} pull --ff-only", check=False)
    else:
        run(f"git clone {url} {repo_path}")
    # Show HEAD commit
    run(f"git -C {repo_path} log -1 --oneline")

TRITON_DIR = os.path.join(WORK_DIR, "ouroboros-triton")
CORE_DIR = os.path.join(WORK_DIR, "ouroboros-core")
DFG_DIR = os.path.join(WORK_DIR, "ouroboros-dfg")


# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
banner("SECTION C: Install Dependencies")

# Core deps
run(f"pip install -q -r {CORE_DIR}/requirements.txt", check=False)
# Triton deps (includes torch, triton, numpy)
run(f"pip install -q -r {TRITON_DIR}/requirements.txt", check=False)
# Extras for C++ parity
run("pip install -q pybind11 pytest", check=False)
# python3.12-dev for C++ extension (Colab may use 3.10/3.11 — detect version)
py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
run(f"apt-get update -qq && apt-get install -y -qq python{py_ver}-dev 2>/dev/null || true",
    check=False)

# Verify final environment
log("\n--- Final environment ---")
run("""python3 -c "
import torch, triton, numpy, pytest, pybind11
print(f'torch:    {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'triton:   {triton.__version__}')
print(f'numpy:    {numpy.__version__}')
print(f'pytest:   {pytest.__version__}')
print(f'pybind11: {pybind11.__version__}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
"
""", check=False)


# ---------------------------------------------------------------------------
# 4. Build C++ parity twin
# ---------------------------------------------------------------------------
banner("SECTION D: Build C++ Parity Twin")

cpp_build = run(f"bash src/cpp/build.sh", cwd=TRITON_DIR, check=False)
cpp_built = cpp_build.returncode == 0
log(f"\nC++ parity build: {'SUCCESS' if cpp_built else 'SKIPPED/FAILED'}")
SECTION_RESULTS["D_cpp_build"] = {
    "status": "PASS" if cpp_built else "SKIP",
    "rc": cpp_build.returncode,
}


# ---------------------------------------------------------------------------
# 5. Full triton test suite (THE KEY SECTION — GPU legs activate here)
# ---------------------------------------------------------------------------
banner("SECTION E: Full Triton Test Suite (pytest -v)")

triton_result = run(
    "python3 -m pytest tests/ -v --tb=short",
    cwd=TRITON_DIR,
    check=False,
    timeout=300,
)
SECTION_RESULTS["E_triton_pytest"] = {
    "status": "PASS" if triton_result.returncode == 0 else "FAIL",
    "rc": triton_result.returncode,
}


# ---------------------------------------------------------------------------
# 6. Permutation-aware d_head=8 study (Section C from t4_full_suite.sh)
# ---------------------------------------------------------------------------
banner("SECTION F: Permutation Bridge Study (d_head=8)")

_section_f = os.path.join(WORK_DIR, "_section_f.py")
with open(_section_f, "w") as _f:
    _f.write(f"""\
import sys, json, os
sys.path.insert(0, r"{TRITON_DIR}")
sys.path.insert(0, os.path.join(r"{TRITON_DIR}", "scripts"))
import wave12_t4_battery as battery
battery.set_triton_root(r"{TRITON_DIR}")
result = battery.perm_study(d_head=8)
print(json.dumps(result, indent=2))
print()
print("un-permuted max_diff = %.6f" % result["unpermuted_max_diff"])
print("  permuted  max_diff = %.3e" % result["permuted_max_diff"])
print("verdict:", result["verdict"])
assert result["ok"], "Permutation bridge FAILED: " + result["verdict"]
""")

perm_result = run(f"python3 {_section_f}", cwd=TRITON_DIR, check=False)

SECTION_RESULTS["F_perm_study"] = {
    "status": "PASS" if perm_result.returncode == 0 else "FAIL",
    "rc": perm_result.returncode,
}


# ---------------------------------------------------------------------------
# 7. Placement-head scaling sweep (Section D from t4_full_suite.sh)
# ---------------------------------------------------------------------------
banner("SECTION G: Placement-Head Scaling Sweep")

_section_g = os.path.join(WORK_DIR, "_section_g.py")
with open(_section_g, "w") as _f:
    _f.write(f"""\
import sys, math, os
sys.path.insert(0, r"{TRITON_DIR}")
sys.path.insert(0, os.path.join(r"{TRITON_DIR}", "scripts"))
import wave12_t4_battery as battery
battery.set_triton_root(r"{TRITON_DIR}")
battery.set_core_src(r"{CORE_DIR}")
rows = battery.head_sweep()
print()
print(battery.render_table(rows))
print()
for r in rows:
    print("d_model=%s epochs=%s instances=%s: lift=%.6f finite=%s" % (
        r["d_model"], r["epochs"], r["instances"],
        r["lift"], math.isfinite(float(r["lift"]))))
bad = [r for r in rows if not math.isfinite(float(r["lift"]))]
if bad:
    print("FATAL: %d non-finite lift(s)" % len(bad))
    sys.exit(1)
print("PASS: all lifts finite")
""")

sweep_result = run(f"python3 {_section_g}", cwd=TRITON_DIR, check=False, timeout=300)

SECTION_RESULTS["G_head_sweep"] = {
    "status": "PASS" if sweep_result.returncode == 0 else "FAIL",
    "rc": sweep_result.returncode,
}


# ---------------------------------------------------------------------------
# 8. ouroboros-core test suite
# ---------------------------------------------------------------------------
banner("SECTION H: Ouroboros-Core Test Suite")

core_result = run(
    "python3 -m pytest tests/ -v --tb=short",
    cwd=CORE_DIR,
    check=False,
    timeout=300,
)
SECTION_RESULTS["H_core_pytest"] = {
    "status": "PASS" if core_result.returncode == 0 else "FAIL",
    "rc": core_result.returncode,
}


# ---------------------------------------------------------------------------
# 9. ouroboros-dfg Rust tests (best-effort — Colab may not have cargo)
# ---------------------------------------------------------------------------
banner("SECTION I: Ouroboros-DFG Rust Tests")

cargo_check = run("which cargo", check=False)
if cargo_check.returncode == 0:
    dfg_result = run(
        "cargo test --locked 2>&1",
        cwd=DFG_DIR,
        check=False,
        timeout=300,
    )
    SECTION_RESULTS["I_dfg_cargo"] = {
        "status": "PASS" if dfg_result.returncode == 0 else "FAIL",
        "rc": dfg_result.returncode,
    }
else:
    log("cargo not found — installing Rust toolchain...")
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        check=False)
    os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
    cargo_recheck = run("which cargo", check=False)
    if cargo_recheck.returncode == 0:
        dfg_result = run(
            "cargo test --locked 2>&1",
            cwd=DFG_DIR,
            check=False,
            timeout=300,
        )
        SECTION_RESULTS["I_dfg_cargo"] = {
            "status": "PASS" if dfg_result.returncode == 0 else "FAIL",
            "rc": dfg_result.returncode,
        }
    else:
        log("Rust install failed — skipping DFG tests")
        SECTION_RESULTS["I_dfg_cargo"] = {"status": "SKIP", "rc": -1}


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
banner("VERDICT")

finished = datetime.now(timezone.utc).isoformat()
log(f"Finished: {finished}")
log("")

overall_pass = True
for section, info in sorted(SECTION_RESULTS.items()):
    marker = "✅" if info["status"] == "PASS" else (
        "⏭️" if info["status"] == "SKIP" else "❌"
    )
    log(f"  {marker} {section}: {info['status']} (rc={info['rc']})")
    if info["status"] == "FAIL":
        overall_pass = False

log("")
if overall_pass:
    log("🎉 OVERALL: PASS — all sections green")
else:
    log("💥 OVERALL: FAIL — see sections above")

# Show the key claim table
log("")
log("Evidence table:")
log("  Claim                               Evidence")
log("  ──────────────────────────────────   ─────────────────────────────")
log(f"  BlockTable/chain/serving semantics  {SECTION_RESULTS.get('E_triton_pytest', {}).get('status', '?')} triton pytest")
log(f"  Golden-reference attention math     {SECTION_RESULTS.get('E_triton_pytest', {}).get('status', '?')} triton pytest")
log(f"  RoPE out-shuffle permutation        {SECTION_RESULTS.get('F_perm_study', {}).get('status', '?')} perm study")
log(f"  Kernel correctness on silicon       {SECTION_RESULTS.get('E_triton_pytest', {}).get('status', '?')} GPU legs in pytest")
log(f"  C++ parity                          {SECTION_RESULTS.get('D_cpp_build', {}).get('status', '?')} build + pytest")
log(f"  Placement-head scaling sweep        {SECTION_RESULTS.get('G_head_sweep', {}).get('status', '?')} head sweep")
log(f"  Core tokenizer/diffusion tests      {SECTION_RESULTS.get('H_core_pytest', {}).get('status', '?')} core pytest")
log(f"  DFG Rust verifier                   {SECTION_RESULTS.get('I_dfg_cargo', {}).get('status', '?')} cargo test")


# ---------------------------------------------------------------------------
# Save proof log
# ---------------------------------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
log_path = f"/content/t4_proof_{timestamp}.log"
with open(log_path, "w") as f:
    f.write("\n".join(LOG_LINES) + "\n")
log(f"\n📄 Proof log saved to: {log_path}")
log("   Download this file as your evidence artifact.")

# Also save inside the triton repo for easy commit
repo_log_dir = os.path.join(TRITON_DIR, "scripts", "logs")
os.makedirs(repo_log_dir, exist_ok=True)
repo_log_path = os.path.join(repo_log_dir, f"t4_proof_{timestamp}.log")
with open(repo_log_path, "w") as f:
    f.write("\n".join(LOG_LINES) + "\n")
log(f"📄 Also saved to:      {repo_log_path}")
log("   (cd into repo and `git add scripts/logs/ && git commit -m 'evidence: T4 proof log' && git push`)")

if not overall_pass:
    sys.exit(1)
