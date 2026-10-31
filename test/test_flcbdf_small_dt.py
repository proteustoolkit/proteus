"""FLCBDF with a tight error tolerance on both the level set and the VOF model must not stall.

The lagged shock capturing of NCLS and VOF used to take the BDF time derivative into its residual.
Per step the level set changes by about 1e-3 however small dt is, so that derivative grows like
1/dt, the viscosity (numDiff * dt ~ const) with it, and the controller kept cutting dt: vortex2D
died at t = 0.9326 with "step size reduced to machine precision" (dt_min only worked around it).
FLCBDF now keeps the time term out of the shock capturing residual.

The run is the hot start measurement harness, test/hotstart_measure, in a subprocess.
"""
import os
import subprocess
import sys

import pytest

h5py = pytest.importorskip("h5py")

here = os.path.dirname(os.path.abspath(__file__))
harness = os.path.join(here, "hotstart_measure")
experimental = os.path.join(os.path.dirname(here), "experimental", "flcbdf")


def test_flcbdf_with_a_tight_tolerance_on_ls_and_vof_passes_the_old_stall_time(tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", HS_TI_LS="flcbdf", HS_TI_VOF="flcbdf",
               HS_FLCBDF_TOL="1e-4", HS_ENDN="12",
               PYTHONPATH=os.pathsep.join([harness, experimental, os.environ.get("PYTHONPATH", "")]))
    env.pop("HS_DTMIN", None)
    run = subprocess.run([sys.executable, os.path.join(harness, "run_case.py"), str(tmp_path), "flcbdf", "1", "full"],
                         env=env, capture_output=True, text=True)
    if "flcbdfWrappers" in run.stderr and "ModuleNotFoundError" in run.stderr:
        pytest.skip("the experimental flcbdf extension is not built")
    assert run.returncode == 0, run.stderr[-2000:]
    # t = 1.2 is past the stall at 0.93; the last archived step exists only if the run got there
    with h5py.File(tmp_path / "hs_flcbdf1.h5", "r") as f:
        assert "u_t12" in f
