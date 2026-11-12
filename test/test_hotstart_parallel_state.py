"""A hot start in parallel restores the lag state, not only the fields.

The shock capturing of the level set and of the volume of fluid lags a step,
and its coefficients live at the quadrature points of each rank's own
elements. Hot started without them, a two-rank run drifts from the one that
was never stopped (1e-3 after one step, 2e-2 after 40 on vortex2D); with them
it agrees to what the linear solver's tolerance allows. The state is written
per rank, so it can only be read back by the same partition: with another
number of ranks the run must say so, restart its history from the solution,
and go on.

The runs use the hot start measurement harness (test/hotstart_measure) on
vortex2D under mpiexec; HS_PARALLEL gives the parallel runs real PETSc solvers
with a tight tolerance so that the comparison is not limited by it.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

here = os.path.dirname(os.path.abspath(__file__))
harness = os.path.join(here, "hotstart_measure")


def _find_mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _find_mpiexec()
pytestmark = pytest.mark.skipif(MPIEXEC is None, reason="needs mpiexec")


def _run(out, ranks, mode, n, endn=None):
    env = dict(os.environ, HS_PARALLEL="1", HS_PARALLEL_ATOL="1e-14", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]))
    if endn is not None:
        env["HS_ENDN"] = str(endn)
    cmd = [MPIEXEC, "-n", str(ranks), sys.executable, os.path.join(harness, "run_case.py"),
           str(out), "be", "1", mode] + ([str(n)] if mode != "full" else [])
    subprocess.run(cmd, env=env, check=True, capture_output=True, timeout=600)


def _final_difference(a, b, step, fields=("u", "vof")):
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        worst = 0.0
        for f in fields:
            x = np.array(fa["%s_t%d" % (f, step)]).ravel()
            y = np.array(fb["%s_t%d" % (f, step)]).ravel()
            worst = max(worst, np.linalg.norm(x - y) / np.linalg.norm(x))
        return worst


def test_two_rank_hot_start_matches_the_run_that_was_not_stopped(tmp_path):
    n, end = 6, 9
    _run(tmp_path / "full", 2, "full", n, endn=end)
    _run(tmp_path / "hot", 2, "stop", n)
    _run(tmp_path / "hot", 2, "hot", n, endn=end)
    h5 = "hs_be1.h5"
    assert _final_difference(tmp_path / "full" / h5, tmp_path / "hot" / h5, end) < 1e-10


def test_hot_start_with_another_number_of_ranks_restarts_and_says_so(tmp_path):
    n, end = 6, 8
    _run(tmp_path / "hot", 2, "stop", n)
    _run(tmp_path / "hot", 1, "hot", n, endn=end)
    log = (tmp_path / "hot" / "proteus_default.log").read_text()
    assert re.search(r"Hot start: no solver state to restore at step \d+ for model \S+ \(none was archived \(a frame carries the solution only[^)]*\), "
                     r"or it was written by a different number of ranks", log)
    with h5py.File(tmp_path / "hot" / "hs_be1.h5", "r") as f:
        assert "u_t%d" % end in f
