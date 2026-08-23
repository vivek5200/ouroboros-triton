#!/usr/bin/env bash
# Build the `ouroboros_cpp` pybind11 extension for the production C++
# BlockTable. Skips with a clear message when g++ or pybind11 are missing.
#
# Usage:  ./src/cpp/build.sh          (or: bash src/cpp/build.sh)
# Output: src/cpp/ouroboros_cpp<ext-suffix>.so   (importable from src/cpp/)
set -u

cd "$(dirname "$0")/.."   # repo root

PYTHON="${PYTHON:-python3}"
OUT_DIR="src/cpp"

echo "=== ouroboros_cpp build ==="

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "SKIP: python interpreter '${PYTHON}' not found"
    exit 0
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "SKIP: g++ not available (install build-essential)"
    exit 0
fi

PYBIND11_INCLUDES="$("$PYTHON" -m pybind11 --includes 2>/dev/null)" || PYBIND11_INCLUDES=""
if [ -z "$PYBIND11_INCLUDES" ]; then
    echo "SKIP: pybind11 not installed (pip install pybind11) — C++ parity tests will skip"
    exit 0
fi

EXT_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="${OUT_DIR}/ouroboros_cpp${EXT_SUFFIX}"

echo "g++     : $(g++ -dumpfullversion -dumpversion 2>/dev/null || g++ -dumpversion)"
echo "python  : $($PYTHON -c 'import sys; print(sys.version.split()[0])')"
echo "output  : $OUT"

# shellcheck disable=SC2086
g++ -O2 -std=c++17 -shared -fPIC $PYBIND11_INCLUDES \
    src/cpp/bindings.cpp src/cpp/block_table.cpp \
    -o "$OUT" || { echo "FAIL: compilation error (see above)"; exit 1; }

echo "OK: built $OUT — tests/test_cpp_parity.py will now run the differential suite"
