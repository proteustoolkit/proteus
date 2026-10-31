"""Tests of the experimental FLCBDF build (skipped when the extension is not built).

    PYTHONPATH=experimental/flcbdf python -m pytest experimental/flcbdf/test_flcbdf.py
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
flcbdfWrappers = pytest.importorskip("flcbdfWrappers")
import ode_check  # noqa: E402


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the integrator writes a data_*.txt log


def test_ode_error_falls_and_steps_grow_with_tolerance():
    results = [ode_check.integrate(rtol) for rtol in (1e-2, 1e-4, 1e-6)]
    errors = [r[0] for r in results]
    steps = [r[1] for r in results]
    assert errors[0] > errors[1] > errors[2]
    assert steps[0] < steps[1] < steps[2]
    assert errors[2] < 1e-8


def test_order_one_with_fixed_steps_is_backward_euler():
    """Order 1 and a constant step is backward Euler. The step is chosen by the
    integrator and then set to another (what a system of models does when
    another model's step is smaller); that used to leave the coefficients of
    the chosen step in place when the set step equalled the previous one."""
    H, lam = 0.1, 1.0
    y = np.array([1.0])
    yp = np.array([-lam])
    D = np.zeros(1)
    ypn = np.zeros(1)
    ynew = np.zeros(1)
    f = flcbdfWrappers.FLCBDF_integrator(y, "be_equivalence")
    f.setTolerances(1.0e-10, 1.0e3, np.ones(1))
    f.set_order(1)
    f.initialize_dt(0.0, 10.0, y, yp)
    f.set_dt(H)
    f.initializeTimeHistory(y, yp)
    for n in range(8):
        f.setInitialGuess(ynew)
        f.calculate_yprime(ynew, D, ypn, D)
        ynew[:] = ynew - (ypn + lam * ynew) / (f.getCurrentAlpha() + lam)
        f.calculate_yprime(ynew, D, ypn, D)
        f.lastStepErrorOk(ynew)
        f.stepTaken(ynew)
        assert abs(ynew[0] - (1.0 / (1.0 + H)) ** (n + 1)) < 1e-13
        f.choose_dt(0.0, 10.0)
        f.set_dt(H)


def test_step_size_at_machine_precision_raises():
    y = np.array([1.0])
    f = flcbdfWrappers.FLCBDF_integrator(y, "tiny_step")
    f.setTolerances(1.0e-10, 1.0e-4, np.ones(1))
    f.initialize_dt(0.0, 10.0, y, np.array([-1.0]))
    f.initializeTimeHistory(y, np.array([-1.0]))
    with pytest.raises(RuntimeError, match="machine precision"):
        for _ in range(2000):
            f.retryStep_errorFailure()


def _vortex(tmp_path, ti_ls, ti_vof, tol, **extra):
    root = os.path.abspath(os.path.join(here, "..", ".."))
    env = dict(os.environ, HS_TI_LS=ti_ls, HS_TI_VOF=ti_vof, HS_FLCBDF_TOL=str(tol), **extra,
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    run = os.path.join(root, "test", "hotstart_measure", "run_case.py")
    out = tmp_path / ("run_%s_%s" % (ti_ls, ti_vof))
    subprocess.run([sys.executable, run, str(out), "flcbdf", "1", "full"],
                   env=env, check=True, capture_output=True)
    log = (out / "proteus_default.log").read_text()
    times = [float(t) for t in re.findall(r"Step Taken, System time step t= *([0-9.e+-]+)", log)]
    return times, log


@pytest.mark.parametrize("ti_ls,ti_vof,tol", [("flcbdf", "be", 1e-2),
                                              ("flcbdf", "be", 1e-4),
                                              ("be", "flcbdf", 1e-4),
                                              ("flcbdf", "flcbdf", 1e-3)])
def test_vortex2d_with_flcbdf_reaches_the_end(tmp_path, ti_ls, ti_vof, tol):
    """vortex2D to t = 8 with FLCBDF on the level set (also at the default
    tolerance 1e-4), on the volume of fluid, and on both, with no Newton
    failure. Both at 1e-4 without a floor is test/test_flcbdf_small_dt.py."""
    times, log = _vortex(tmp_path, ti_ls, ti_vof, tol)
    assert times and times[-1] >= 8.0 - 1e-8
    assert "System Step Failed" not in log


def test_dt_floor_gets_both_models_at_1e_4_past_the_lagged_shock_capturing_collapse(tmp_path):
    """The dt floor, which used to be the only way past the collapse of both models at tol 1e-4 at
    t = 0.93 (the lagged shock capturing fed on the BDF time derivative; FLCBDF no longer puts it
    in that residual, see test/test_flcbdf_small_dt.py). With dt_min the step stays at the floor
    and the error test is accepted there."""
    times, log = _vortex(tmp_path, "flcbdf", "flcbdf", 1e-4, HS_DTMIN="3e-4", HS_ENDN="11")
    assert times and times[-1] >= 1.1 - 1e-8
    assert "System Step Failed" not in log


def test_hotstart_with_flcbdf_reproduces_the_uninterrupted_run(tmp_path):
    """FLCBDF on both models: stop at output 10, hot start from the archive,
    and the fields at the end agree with the run that was never stopped. The
    archive carries each integrator's history, order and step size
    (FLCBDF.state / set_state); the step controllers follow it
    (FLCBDF_controller.syncAfterRestore)."""
    root = os.path.abspath(os.path.join(here, "..", ".."))
    hm = os.path.join(root, "test", "hotstart_measure")
    env = dict(os.environ, HS_FLCBDF_TOL="1e-2",
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    def run(out, *mode):
        subprocess.run([sys.executable, os.path.join(hm, "run_case.py"), str(tmp_path / out), "flcbdf", "1", *mode],
                       env=env, check=True, capture_output=True)
    run("full", "full")
    run("hot", "stop", "10")
    run("hot", "hot", "10")
    out = subprocess.run([sys.executable, os.path.join(hm, "compare.py"),
                          str(tmp_path / "full" / "hs_flcbdf1.h5"), str(tmp_path / "hot" / "hs_flcbdf1.h5"),
                          "flcbdf1", "10"], env=env, check=True, capture_output=True, text=True).stdout
    final = [l for l in out.splitlines() if l.startswith("FINAL")][0]
    errs = [float(x) for x in re.findall(r"(?:L2|Linf)=([0-9.e+-]+)", final)]
    assert len(errs) == 8 and max(errs) < 1e-10, final


def _ode_parallel(n):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run(["mpiexec", "-n", str(n), sys.executable, os.path.join(here, "ode_parallel.py")],
                         env=env, check=True, capture_output=True, text=True).stdout
    return __import__("json").loads([l for l in out.splitlines() if l.startswith("{")][0])


def test_step_sequence_does_not_depend_on_the_number_of_ranks():
    """The error norm is global: one rank and two (unequal) ranks take the same steps."""
    one, two = _ode_parallel(1), _ode_parallel(2)
    assert len(one["dts"]) == len(two["dts"]) and len(one["dts"]) > 10
    assert np.allclose(one["dts"], two["dts"], rtol=1e-10, atol=0.0)
    assert abs(one["err"] - two["err"]) < 1e-10


def test_vortex2d_with_flcbdf_on_two_ranks_reaches_the_end(tmp_path):
    """Both models on FLCBDF, two ranks: the run completes and takes about as many steps
    as one rank (the error norm counts each element once, ghosts excluded)."""
    root = os.path.abspath(os.path.join(here, "..", ".."))
    env = dict(os.environ, HS_PARALLEL="1", HS_TI_LS="flcbdf", HS_TI_VOF="flcbdf", HS_FLCBDF_TOL="1e-3",
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    run = os.path.join(root, "test", "hotstart_measure", "run_case.py")
    n = {}
    for ranks in (1, 2):
        out = tmp_path / ("ranks%d" % ranks)
        subprocess.run(["mpiexec", "-n", str(ranks), sys.executable, run, str(out), "flcbdf", "1", "full"],
                       env=env, check=True, capture_output=True)
        log = (out / "proteus_default.log").read_text()
        times = [float(t) for t in re.findall(r"Step Taken, System time step t= *([0-9.e+-]+)", log)]
        assert times and times[-1] >= 8.0 - 1e-8 and "System Step Failed" not in log
        n[ranks] = len(times)
    assert abs(n[2] - n[1]) < 0.1 * n[1]


def _two_rank_run(tmp_path, tol):
    """Both models on FLCBDF, run under mpiexec by the hot start measurement harness. Returns the
    runner (output directory, ranks, harness arguments), the environment and the harness directory."""
    root = os.path.abspath(os.path.join(here, "..", ".."))
    hm = os.path.join(root, "test", "hotstart_measure")
    env = dict(os.environ, HS_FLCBDF_TOL=tol, HS_PARALLEL="1", HS_PARALLEL_ATOL="1e-14",
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    def run(out, ranks, *mode):
        return subprocess.run(["mpiexec", "-n", str(ranks), sys.executable, os.path.join(hm, "run_case.py"),
                               str(tmp_path / out), "flcbdf", "1", *mode], env=env, check=True,
                              capture_output=True, text=True)
    return run, env, hm


@pytest.mark.parametrize("tol", ["1e-2", "1e-4"])
def test_hotstart_with_flcbdf_on_two_ranks_reproduces_the_uninterrupted_run(tmp_path, tol):
    """FLCBDF on both models, two ranks: each rank archives its own copy of every integrator's
    state (solver_state/..._p<rank>) and the hot start from it, on the same partition, ends where
    the run that was never stopped does. At 1e-4 the models take many more steps, and used to
    stall at t = 0.93 (see test/test_flcbdf_small_dt.py)."""
    run, env, hm = _two_rank_run(tmp_path, tol)
    run("full", 2, "full")
    run("hot", 2, "stop", "10")
    run("hot", 2, "hot", "10")
    out = subprocess.run([sys.executable, os.path.join(hm, "compare.py"),
                          str(tmp_path / "full" / "hs_flcbdf1.h5"), str(tmp_path / "hot" / "hs_flcbdf1.h5"),
                          "flcbdf1", "10"], env=env, check=True, capture_output=True, text=True).stdout
    final = [l for l in out.splitlines() if l.startswith("FINAL")][0]
    errs = [float(x) for x in re.findall(r"(?:L2|Linf)=([0-9.e+-]+)", final)]
    assert len(errs) == 8 and max(errs) < 1e-10, final


def test_hotstart_with_flcbdf_from_another_number_of_ranks_restarts_the_history_and_finishes(tmp_path):
    """The state is per rank and tied to the partition that wrote it: stopped on two ranks and
    hot started on one, no model finds its state, each says so, rebuilds its history from the
    solution (a restart), and the run goes on to the end. It does not hang and it does not read the
    wrong rank's arrays."""
    run, env, hm = _two_rank_run(tmp_path, "1e-2")
    run("hot", 2, "stop", "10")
    run("hot", 1, "hot", "10")
    log = (tmp_path / "hot" / "proteus_default.log").read_text()
    assert "no solver state to restore at step 10" in log
    assert "System Step Failed" not in log
    import h5py
    with h5py.File(tmp_path / "hot" / "hs_flcbdf1.h5", "r") as f:
        assert "u_t80" in f
