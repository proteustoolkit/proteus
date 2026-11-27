"""SW2D, the Galerkin shallow water kernel, builds a model, steps it, and gives the serial answer on two ranks.

Nothing else in the tree builds ``proteus.kernels.SW2D`` (the shallow water decks use SW2DCV and GN_SW2DCV), so before
this file its 718 statements ran only on import. ``try_supg_stabilization`` on the numerics options selects the SUPG
residual and Jacobian (``calculateResidual_supg``, ``calculateJacobian_supg``) over the plain Galerkin ones. The helper
(sw2d_galerkin_mpi.py) runs a 1.0 / 0.5 depth step on the unit square for three fixed steps with the flat bottom the
coefficients give (no bathymetry), u = v = 0 on the walls, and the tests check:

- the entry points the model bound are the ones the option names (a mutation that swaps them fails here);
- the answer is finite, the depth stays within 2% of its initial range, water flows from the deep to the shallow
  side, and the two variants differ, so the SUPG branch ran;
- two ranks give the serial answer (to 1e-9 SUPG, measured 2e-13; 1e-8 plain, measured 5e-10).

Until the C0 kernel-bugs round the module could not be built (nine names used and never imported) and the helper put the
names in its namespace; it also gave u = v = 0 at every node, interior included, so the plain run did not move. Both
are fixed: the helper pins the velocity on the walls only and the tests check that the dam breaks.
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "sw2d_galerkin_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [HERE, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER, "-l1"] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER, "-l1"]
    launch_output.run(argv, cwd=cwd, env=env, timeout=900)
    return numpy.load(os.path.join(str(cwd), "sw2d.npz"))


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    out = {}
    for supg in (0, 1):
        d = tmp_path_factory.mktemp("serial%d" % supg)
        r = _run(1, d, SUPG=supg)
        out[supg] = {k: r[k] for k in r.files}
    return out


def test_the_option_selects_the_supg_entry_points(serial):
    assert str(serial[0]["residual"]) == "calculateResidual" and str(serial[0]["jacobian"]) == "calculateJacobian"
    assert str(serial[1]["residual"]) == "calculateResidual_supg" and str(serial[1]["jacobian"]) == "calculateJacobian_supg"
    assert serial[0]["residual_calls"] > 0 and serial[1]["residual_calls"] > 0


def test_both_variants_step_and_differ(serial):
    plain, supg = serial[0], serial[1]
    assert plain["h"].size == 121
    for r in (plain, supg):
        for k in ("h", "hu", "hv"):
            assert numpy.isfinite(r[k]).all(), k
    for r in (plain, supg):
        assert 0.48 < r["h"].min() and r["h"].max() < 1.02
        # the dam breaks: water flows from the deep side (x < 0.5) to the shallow one; u (stored as "hu") is the
        # x velocity, v (stored as "hv") stays small (the walls are y = 0, 1 and the step does not depend on y)
        assert r["hu"].max() > 0.1 and r["hu"].min() > -0.05
        assert numpy.abs(r["hv"]).max() < 0.1 * r["hu"].max()
        assert r["h"][r["x"] < 0.4].min() < 1.0 - 1e-4, "the deep side did not drain"
    assert numpy.abs(plain["h"] - supg["h"]).max() > 1e-4, "the SUPG variant gave the plain answer"


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("supg", [0, 1])
def test_two_ranks_give_the_serial_answer(serial, tmp_path, supg):
    two = _run(2, tmp_path, SUPG=supg)
    one = serial[supg]
    assert numpy.array_equal(one["x"], two["x"]) and numpy.array_equal(one["y"], two["y"])
    for k in ("h", "hu", "hv"):
        err = numpy.abs(one[k] - two[k]).max()
        # the plain Newton crawls (its residual recomputes the viscosity the Jacobian holds fixed): 5e-10 measured on u
        tol = 1e-8 if supg == 0 else 1e-9
        assert err < tol, (k, "two ranks differ from serial by %.3e" % err)
