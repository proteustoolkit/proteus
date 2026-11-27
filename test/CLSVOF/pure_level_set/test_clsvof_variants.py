"""C49 R4: CLSVOF options that no deck in the suite selects, on the 2D solid-body rotation of test_clsvof.py (case 2).

The deck keeps its CLSVOF settings as module variables of ``clsvof.py`` (``computeMetrics``, ``doSpinUpStep``,
``useMetrics``, ``lambdaFact``); test_clsvof.py runs it with the defaults.  Here each option is switched on in turn
and what it must do is checked:

* ``computeMetrics`` 1, 2 and 3 only write diagnostics: the solution is bit-identical to the default run, and the files
  have the documented columns, one row per step (ETS) or one row at the end (EOS), with a volume error that is small;
* ``doSpinUpStep`` replaces the initial condition by its L2 projection: the run starts from a different field and the
  result stays close to the default;
* ``useMetrics`` (anisotropic metric in the regularization) and ``lambdaFact`` change the solution slightly and keep it
  bounded; the level set stays a sign function of the interface (values within [-1, 1] up to a small excess).
"""
import os
import re
from importlib import reload

import numpy as np
import pytest

import proteus
from proteus import default_n, default_s, default_so

from . import clsvof, clsvof_n, clsvof_p, parameters

VARIANTS = {
    "default": dict(),
    "metrics1": dict(computeMetrics=1),
    "metrics3": dict(computeMetrics=3),
    "useMetrics": dict(useMetrics=1),
    "lambda": dict(lambdaFact=0.5),
}
_cache = {}


def _run(opts, directory, name, **options):
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        default_n.conservativeFlux = None
        parameters.ct.test_case = 2
        reload(clsvof)
        for k, v in options.items():
            setattr(clsvof, k, v)
        reload(clsvof_p)
        reload(clsvof_n)
        so = default_so
        so.tnList = clsvof_n.tnList
        clsvof_p.name = "clsvof_" + name
        so.name = "clsvof_" + name
        ns = proteus.NumericalSolution.NS_base(so, [clsvof_p], [clsvof_n], [default_s], opts)
        ns.calculateSolution(so.name)
        m = ns.modelList[0].levelModelList[-1]
        u = np.array(m.u[0].dof)
        u0 = np.array(m.u0_dof) if hasattr(m, "u0_dof") else None
        files = {f: open(f).read() for f in os.listdir(".") if f.endswith(".csv")}
    finally:
        os.chdir(cwd)
    return u, u0, files


@pytest.fixture
def run(opts, tmp_path_factory):
    def go(name):
        if name not in _cache:
            d = tmp_path_factory.mktemp("clsvof_" + name)
            _cache[name] = _run(opts, str(d), name, **VARIANTS[name])
        return _cache[name]
    return go


def _numbers(line):
    """the floats in one csv line, whatever repr() wrote (numpy 2 writes 'np.float64(1.5)')"""
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", re.sub(r"np\.float64", "", line))]


def _metrics(files, kind):
    name = [f for f in files if f.endswith(kind + ".csv")]
    assert len(name) == 1, sorted(files)
    return files[name[0]].strip().split("\n")


@pytest.mark.parametrize("name", ["metrics1", "metrics3"])
def test_metrics_only_write_diagnostics_the_solution_is_unchanged(run, name):
    np.testing.assert_array_equal(run(name)[0], run("default")[0])
    assert run("default")[2] == {}                         # and a default run writes no metrics file


def test_metrics_at_every_time_step_report_a_tiny_volume_error(run):
    lines = _metrics(run("metrics1")[2], "metricsAtETS")
    assert lines[0] == "time,global_sV_err"
    assert len(lines) > 7                                   # one row per time step
    errs = [_numbers(l)[-1] for l in lines[1:]]
    # the smoothed-volume error starts at round-off and accumulates, never beyond the nonlinear tolerance of the run (1e-6 here)
    assert all(0. <= e < 1e-6 for e in errs), errs
    assert errs[0] < 1e-10 < errs[-1]                      # it accumulates from round-off to a plateau (5e-8)


def test_metrics_at_the_end_report_the_error_against_the_exact_rotated_disk(run):
    lines = _metrics(run("metrics3")[2], "metricsAtEOS")
    assert lines[0].split(",") == ["global_I_err", "global_sI_err", "global_V_err", "global_sV_err", "global_D_err",
                                   "global_L2_err", "global_L2Banded_err", "global_sH_L2_err"]
    assert len(lines) == 2
    I, sI, V, sV, D, L2, L2b, sH = _numbers(lines[1])
    # the smoothed volume is conserved to the nonlinear tolerance; the others measure the coarse 21 x 21 mesh
    assert sV < 1e-6
    assert 0. < L2 < 0.15 and 0. < L2b < 0.15 and 0. < D < 0.05 and 0. < I < 0.15 and 0. < V < 0.15
    np.testing.assert_allclose([I, V, D, L2], [0.0767448, 0.1000906, 0.0114262, 0.0996349], rtol=1e-3)


def _model(opts, directory, name, **options):
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        default_n.conservativeFlux = None
        parameters.ct.test_case = 2
        reload(clsvof)
        for k, v in options.items():
            setattr(clsvof, k, v)
        reload(clsvof_p)
        reload(clsvof_n)
        so = default_so
        so.tnList = clsvof_n.tnList
        clsvof_p.name = so.name = "clsvof_" + name
        ns = proteus.NumericalSolution.NS_base(so, [clsvof_p], [clsvof_n], [default_s], opts)
        return ns, ns.modelList[0].levelModelList[-1]
    finally:
        os.chdir(cwd)


def test_the_spin_up_system_is_the_l2_projection_of_the_initial_condition(opts, tmp_path):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    ns, m = _model(opts, str(tmp_path), "spinup_system", doSpinUpStep=True)
    assert m.uInitial is not None and m.spinUpStepTaken is False
    r = np.zeros(m.dim)
    m.assembleSpinUpSystem(r, m.jacobian)
    rowptr, colind, nzval = m.jacobian.getCSRrepresentation()
    M = csr_matrix((nzval, colind, rowptr), shape=(m.dim, m.dim))
    # the mass matrix: its entries add up to the area of the unit square
    np.testing.assert_allclose(M.sum(), 1.0, rtol=1e-12)
    # the right-hand side integrates the initial condition: its sum is the integral of u0 by the same quadrature
    # the projection M u = r reproduces the initial condition: the same field, to the projection error of P1
    u = spsolve(M.tocsc(), r)
    u0 = np.array([clsvof_p.initialConditions[0].uOfXT(x, 0.) for x in m.mesh.nodeArray])
    assert np.abs(u - u0).max() < 0.06 * np.abs(u0).max()
    # the integral of the initial condition (the sum of the right-hand side) is that of its P1 interpolant, to the
    # quadrature error of the interpolation
    np.testing.assert_allclose(r.sum(), np.asarray(M.sum(axis=0)).ravel() @ u0, rtol=0.05)
    np.testing.assert_allclose(M @ u, r, atol=1e-12)


@pytest.mark.xfail(strict=True, reason="CLSVOFNewton.solve never calls spinUpStep (the only caller, solveOldMethod, "
                                       "was deleted by C49 closeout; nothing called it): doSpinUpStep=True changes "
                                       "nothing in a run (found by C49 R4; kernel territory: C0)")
def test_doSpinUpStep_takes_the_spin_up_step_in_a_run(opts, tmp_path):
    from proteus.kernels import CLSVOF
    calls = []
    original = CLSVOF.LevelModel.assembleSpinUpSystem
    CLSVOF.LevelModel.assembleSpinUpSystem = lambda self, r, j: (calls.append(1), original(self, r, j))[1]
    try:
        _run(opts, str(tmp_path), "spinup_run", doSpinUpStep=True)
    finally:
        CLSVOF.LevelModel.assembleSpinUpSystem = original
    assert sum(calls) == 1


def test_use_metrics_and_lambda_factor_change_the_solution_and_keep_it_bounded(run):
    default = run("default")[0]
    um, lam = run("useMetrics")[0], run("lambda")[0]
    assert np.abs(um - default).max() < 1e-8              # isotropic mesh: the metric reduces to the identity
    assert np.abs(lam - default).max() > 1e-3             # a weaker regularization is a different solution
    assert np.linalg.norm(lam - default) < 0.2 * np.linalg.norm(default)
    for u in (um, lam):
        assert np.all(np.isfinite(u)) and u.min() > -1.0 and u.max() < 0.2


@pytest.mark.xfail(strict=True, raises=NameError,
                   reason="CLSVOF.py uses math.sqrt in preStep (computeMetrics=2, the per-step metrics) without "
                          "importing math: the option cannot run (found by C49 R4; kernel territory: C0)")
def test_metrics_2_per_step_diagnostics_run(opts, tmp_path):
    u, _, files = _run(opts, str(tmp_path), "metrics2", computeMetrics=2)
    assert len(_metrics(files, "metricsAtETS")) > 1


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="CLSVOF writes the metrics files with repr(): under numpy 2 the values are "
                          "'np.float64(1.5e-12)' (not a number to the next reader) and the time column is "
                          "repr(t)[:4] ('1e-0' for t = 1e-05) (found by C49 R4; kernel territory: C0)")
def test_the_per_step_metrics_file_is_plain_csv_with_the_time_of_each_step(run):
    lines = _metrics(run("metrics1")[2], "metricsAtETS")
    assert lines[0] == "time,global_sV_err"
    times = [float(l.split(",")[0]) for l in lines[1:]]
    values = [float(l.split(",")[1]) for l in lines[1:]]
    assert times == sorted(times) and times[-1] == pytest.approx(0.1, rel=1e-6)
    assert max(values) < 1e-6
