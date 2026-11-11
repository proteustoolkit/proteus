#!/usr/bin/env bash
# Run the test suite the way .github/workflows/sdist.yml does, so a local result can be compared with CI's.
#
#   test/run_like_ci.sh [--inplace] [--workers N] [pytest arguments ...]
#
# With no pytest arguments it runs what CI's main suite runs: every test under test/, minus the manuscript decks
# and the two FSI IBM simulations. Arguments (node ids, -k, -x, ...) replace the default target.
#
# What a local run needs to agree with CI, in the order it usually goes wrong:
#
#  1. The environment. CI builds a conda environment from environment-openmpi-dev.yml (Linux) or
#     environment-mpich-dev.yml (macOS arm64). A different PETSc changes results: without MUMPS about 19 tests fail
#     (test_adr_problem's null-space, periodic and Taylor-Hood cases, test_pro_cli, test_rank_interface_flux);
#     another ParMETIS/MPI build partitions differently, and the multilevel dgp1 cases on 3 ranks then find a rank
#     with no elements. That is most of the fleet nightly's 7-31 failures on commits CI passes (pip-built PETSc
#     3.26.0, Python 3.10/3.12). Make the environment with
#        conda env create -n proteus-ci -f environment-openmpi-dev.yml     # or -mpich- on a Mac
#     and build proteus into it (pip install . , or build_ext --inplace and --inplace here).
#  2. ymf at proteustoolkit/ymf@main, which is what the environment files install. An older ymf fails hundreds of
#     archive tests with KeyError/YmfArchiveError. This script prints the ymf commit and the one at main.
#  3. The environment's own mpiexec first on PATH (a system or Homebrew MPI launching the env's mpi4py hangs or
#     aborts); FI_PROVIDER=tcp for conda-forge's MPICH (its libfabric "sockets" provider busy-polls three cores per
#     process); OMPI_MCA_pml=ob1 for Open MPI; one BLAS/OpenMP thread per process.
#  4. A scratch working directory (tests write files under fixed names) and --import-mode=importlib, as CI.
#
# With --inplace (proteus from this checkout, not installed) the scripts setup.py installs (parun, ...) come from this
# checkout too, through wrappers first on PATH, not from the environment's bin. test_import_hygiene also checks 7 files that are in the
# checkout but not in an installed package (proteus/tests/richards, a TwoPhaseFlow documentation example), so the run
# has CI's tests plus those 7.
#
# CI is still the oracle: this reproduces its invocation, not its machine.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPLACE=0
WORKERS=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --inplace) INPLACE=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) break ;;
  esac
done

PY="$(command -v python)"
BIN="$(dirname "$PY")"
export PATH="$BIN:$PATH"
[ "$INPLACE" = 1 ] && export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ "$INPLACE" = 1 ]; then
  # The scripts setup.py installs (parun, povgen.py, ...) are copies in the environment's bin from whatever build was
  # installed there last, and tests launch them by name: a 10-25 parun with no --hotStartMode failed every hot start
  # on 11-05. Put this checkout's copies first on PATH, installed as pip does: a python shebang becomes this
  # environment's python. (Copies, not shell wrappers: the *_mpi.py helpers runpy.run_path(which('parun')).)
  SCRIPTS="$(mktemp -d "${TMPDIR:-/tmp}/proteus-scripts.XXXXXX")"
  "$PY" - "$REPO" "$SCRIPTS" <<'PYEOF' || exit 1
import ast, os, re, sys
repo, out = sys.argv[1], sys.argv[2]
listed = re.search(r"scripts\s*=\s*(\[[^\]]*\])", open(os.path.join(repo, "setup.py")).read())
for rel in ast.literal_eval(listed.group(1)):
    with open(os.path.join(repo, rel), "rb") as f:
        text = f.read()
    first, nl, rest = text.partition(b"\n")
    if first.startswith(b"#!") and b"python" in first:
        text = b"#!" + sys.executable.encode() + nl + rest
    path = os.path.join(out, os.path.basename(rel))
    with open(path, "wb") as f:
        f.write(text)
    os.chmod(path, 0o755)
PYEOF
  export PATH="$SCRIPTS:$PATH"
fi

export MPLBACKEND=AGG OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export PROTEUS_TEST_TIMEOUT_SCALE="${PROTEUS_TEST_TIMEOUT_SCALE:-auto}"
export OMPI_MCA_pml=ob1
if "$BIN/mpiexec" --version 2>/dev/null | grep -qi "hydra\|mpich"; then
  export FI_PROVIDER=tcp
fi

echo "python:   $PY"
"$PY" - <<'EOF'
import importlib.metadata as md, json, sys
try:
    import proteus
    print("proteus:  %s (%s)" % (proteus.__file__, getattr(proteus, "__version__", "?")))
except Exception as exc:
    print("proteus:  NOT IMPORTABLE: %r" % (exc,)); sys.exit(1)
try:
    import os, subprocess, ymf
    here = os.path.dirname(ymf.__file__)
    commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not commit:      # an installed ymf: the commit pip recorded
        raw = md.distribution("ymf").read_text("direct_url.json")
        commit = ((json.loads(raw) if raw else {}).get("vcs_info") or {}).get("commit_id", "(not a git install)")
    print("ymf:      %s at %s" % (ymf.__file__, commit))
except Exception as exc:
    print("ymf:      unknown (%r)" % (exc,))
try:
    import petsc4py
    petsc4py.init([])
    from petsc4py import PETSc
    have = [p for p in ("mumps", "superlu_dist", "parmetis", "ptscotch", "hypre") if PETSc.Sys.hasExternalPackage(p)]
    print("petsc:    %s with %s" % (".".join(map(str, PETSc.Sys.getVersion())), ", ".join(have)))
    if "mumps" not in have:
        print("WARNING:  this PETSc has no MUMPS; about 19 tests that CI passes will fail")
except Exception as exc:
    print("petsc:    unknown (%r)" % (exc,))
EOF
[ $? = 0 ] || exit 1
MAIN_YMF="$(git ls-remote https://github.com/proteustoolkit/ymf.git refs/heads/main 2>/dev/null | cut -c1-40)"
[ -n "$MAIN_YMF" ] && echo "ymf@main: $MAIN_YMF"
echo "mpiexec:  $(command -v mpiexec)   FI_PROVIDER=${FI_PROVIDER:-unset}"
echo "parun:    $(command -v parun)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/proteus-tests.XXXXXX")"
echo "work dir: $WORK"
cd "$WORK" || exit 1

if [ $# -gt 0 ]; then
  TARGET=("$@")
else
  # the same lists as sdist.yml's "Run test suite" step
  TARGET=(
    --ignore="$REPO/test/richards/bio2d" --ignore="$REPO/test/richards/bio2d_FCT"
    --ignore="$REPO/test/richards/Raingarden" --ignore="$REPO/test/richards/Tracy_convergence"
    --ignore="$REPO/test/richards/WRR_input_files" --ignore="$REPO/test/richards/WRR_draft"
    --ignore="$REPO/test/salinity/vdft_reproduce"
    --deselect=test/FSI/test_FSI.py::TestIBM::test_fallingCylinderIBM_sdf
    --deselect=test/FSI/test_FSI.py::TestIBM::test_fallingCylinderIBM_ball
    "$REPO/test"
  )
fi
exec "$PY" -m pytest -v -n "$WORKERS" --dist load --import-mode=importlib --durations=30 -p no:cacheprovider "${TARGET[@]}"
