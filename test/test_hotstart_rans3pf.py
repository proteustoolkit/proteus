"""Hot start of the RANS3PF cylinder case (test/HotStart_3P, momentum + pressure increment + pressure
models) continues the uninterrupted run bit for bit. What it takes: the pressure model's lagged pressure
and sharp pressure and the momentum model's entropy residual, previous solutions and velocity at the
quadrature points (the coefficients owner of the solver_state archive), RANS3PF's lagged shock capturing
(``numDiff_last`` and its state machine), and the stepping loop's first-step flag off, because the
history is back.

Without them the hot start rebuilds all of it from the restart state and drifts (u 1.7e-4 relative at
step 40; the harness is test/hotstart_measure/run_rans3p.py, whose README section has the oracle)."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

here = os.path.dirname(os.path.abspath(__file__))
driver = os.path.join(here, "hotstart_measure", "run_rans3p.py")
STEPS, STOP = 14, 8


def _run(outdir, mode, extra=None):
    env = dict(os.environ, HS_NSTEPS=str(STEPS), MPLBACKEND="AGG", FI_PROVIDER="tcp", OMP_NUM_THREADS="1")
    env.update(extra or {})
    args = [sys.executable, driver, str(outdir), mode] + ([str(STOP)] if mode != "full" else [])
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def _u(outdir, step):
    with h5py.File(os.path.join(str(outdir), "hs_r3pf.h5"), "r") as f:
        return np.array(f["u_t%d" % step]).ravel()


def _error(tmp_path, extra=None):
    _run(tmp_path / "full", "full")
    _run(tmp_path / "hot", "stop", extra)
    _run(tmp_path / "hot", "hot", extra)
    x, y = _u(tmp_path / "full", STEPS), _u(tmp_path / "hot", STEPS)
    return np.linalg.norm(x - y) / np.linalg.norm(x)


def test_hotstart_with_pressure_state_continues_the_uninterrupted_run(tmp_path):
    assert _error(tmp_path) == 0.0  # bit-identical: the archive carries everything the step reads


def test_hotstart_without_the_coefficients_owner_drifts(tmp_path):
    """The check that the test above can fail: the archive as it was before the coefficients owner."""
    assert _error(tmp_path, {"HS_NO_COEF": "1"}) > 1e-5
