#!/usr/bin/env bash
# Build proteus's sdist and check it; nothing is compiled, installed or uploaded.
#
#   tools/check_sdist.sh [OUTDIR]       # default: a new temporary directory
#   EXPECT_VERSION=2.0.0rc1 tools/check_sdist.sh   # also assert the version
#
# Needs the build requirements already installed (pyproject.toml's
# [build-system].requires: setuptools>=61, cython, pybind11, numpy, mpi4py),
# plus `build` and `twine`; it builds with --no-isolation like sdist.yml does.
# That sdist installs and runs in sdist.yml (every leg), so this script does not
# repeat that; it checks the file PyPI would receive: size (PyPI's limit is
# 100 MB per file), content, metadata, and `twine check --strict`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$(mktemp -d)}"
LIMIT_MB="${PROTEUS_SDIST_LIMIT_MB:-50}"      # well under PyPI's 100 MB
mkdir -p "$OUT"
PY="${PYTHON:-python}"

"$PY" -m build --sdist --no-isolation --outdir "$OUT/dist" "$ROOT"
"$PY" -m twine check --strict "$OUT"/dist/*

SDIST="$(ls "$OUT"/dist/py_proteus-*.tar.gz)"
BYTES=$(wc -c < "$SDIST")
echo "sdist: $SDIST ($(( BYTES / 1024 )) KB, limit ${LIMIT_MB} MB)"
[ "$BYTES" -le $(( LIMIT_MB * 1024 * 1024 )) ] || { echo "sdist too large"; exit 1; }

# Compiled or generated files, logs, data and CI files must not ship by accident.
if tar tzf "$SDIST" | grep -E '\.(so|o|pyc|log|h5|xmf|smb|dmg|msh)$|/\.github/|\.DS_Store|/build/'; then
  echo "unexpected files in the sdist (listed above)"; exit 1
fi
# The files a build cannot do without.
for need in setup.py pyproject.toml LICENSE.md README.md PKG-INFO proteus/__init__.py proteus/proteus_blas.h; do
  tar tzf "$SDIST" | grep -q "^[^/]*/$need$" || { echo "missing from the sdist: $need"; exit 1; }
done

TOP="$(tar tzf "$SDIST" | head -1 | cut -d/ -f1)"
INFO="$(tar xzf "$SDIST" -O "$TOP/PKG-INFO")"
NAME="$(printf '%s\n' "$INFO" | sed -n 's/^Name: //p')"
VER="$(printf '%s\n' "$INFO" | sed -n 's/^Version: //p')"
[ "$NAME" = "py-proteus" ] || { echo "Name is '$NAME', not py-proteus"; exit 1; }
printf '%s\n' "$INFO" | grep -q '^Requires-Dist: ymf$' || { echo "ymf is not a requirement"; exit 1; }
if [ -n "${EXPECT_VERSION:-}" ] && [ "$VER" != "$EXPECT_VERSION" ]; then
  echo "built version $VER, expected $EXPECT_VERSION"; exit 1
fi
echo "ok: py-proteus $VER, twine check, size, contents, metadata"
echo "$OUT"
