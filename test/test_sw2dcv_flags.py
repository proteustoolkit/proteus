"""SW2DCV switches no deck sets: LOW_ORDER_ONLY, NO_STAR_STATES, WALL_RIEMANN, NODAL_H_BOUNDS, no bounds relaxation, linear friction,
a tailwater, a walls function, a porosity with a drag law, the consistent mass matrix (KSP/LU solves, a getJacobian), and the dispersive model GN_SW2DCV with the lumped and the consistent mass matrix.

sw2dcv_flags_mpi.py runs the circular_damBreak deck (SWEs, lumped mass matrix, unstructured mesh he = 0.6, t = 0.3) with one
of them on, serially and on two ranks (partition pinned to a slanted cut). Checked, per variant:

- the run is finite and the depth non-negative (to roundoff);
- on a closed domain the water volume (sum of ML h, ML carrying the porosity) is conserved to 1e-11 relative; with a tailwater (open
  edge, stage 0.9 below the still depth 1.0) it is not: the volume falls;
- the answer differs from the default's by more than 1e-8 where the switch changes the scheme (LOW_ORDER_ONLY,
  BOUNDS_RELAXATION 0, LINEAR_FRICTION, tailwater, drag law, consistent mass matrix, GN against SWEs), so the branch ran; NO_STAR_STATES, WALL_RIEMANN, NODAL_H_BOUNDS and the
  walls function give the default's answer here to the last bit (the dam does not reach a wall in 0.3 s and the states stay smooth), so
  for them the run, the conservation and the two-rank comparison are what is checked;
- two ranks give the serial answer to 1e-9;
- (serial, SWEs and GN) ``getJacobian`` is the mass matrix of every unknown: symmetric, total nc times the area.
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
HELPER = os.path.join(HERE, "sw2dcv_flags_mpi.py")

CLOSED = ["default", "low_order_only", "no_star_states", "wall_riemann", "nodal_h_bounds", "no_relaxation",
          "linear_friction", "walls_function", "drag_law", "consistent", "gn", "gn_consistent"]
DIFFERS = ["low_order_only", "no_relaxation", "linear_friction", "tailwater", "drag_law", "consistent", "gn"]
VARIANTS = CLOSED + ["tailwater"]


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, variant):
    env = dict(os.environ, OMP_NUM_THREADS="1", VARIANT=variant)
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [HERE, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    u = numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)
    v0, v1 = [float(x) for x in open(os.path.join(str(cwd), "total.txt")).read().split()]
    return u, v0, v1


def _jacobian(cwd):
    import json
    return json.load(open(os.path.join(str(cwd), "jacobian.json")))


JACOBIANS = {}


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    out = {}
    for variant in VARIANTS:
        d = tmp_path_factory.mktemp("serial_" + variant)
        out[variant] = _run(1, d, variant)
        if variant in ("default", "gn"):
            JACOBIANS[variant] = _jacobian(d)
    return out


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_variant_runs_and_keeps_the_water_volume(serial, variant):
    u, v0, v1 = serial[variant]
    assert numpy.isfinite(u).all()
    assert u[:, 2].min() > -1e-10
    if variant == "tailwater":
        assert v1 < v0 - 1.0, "water leaves through the tailwater edge (stage 0.9 < depth 1.0)"
    else:
        assert abs(v1 - v0) < 1e-11 * v0


@pytest.mark.parametrize("variant", DIFFERS)
def test_the_variant_changes_the_answer(serial, variant):
    assert numpy.abs(serial[variant][0][:, 2:5] - serial["default"][0][:, 2:5]).max() > 1e-8


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant", VARIANTS)
def test_two_ranks_give_the_serial_answer(serial, tmp_path, variant):
    two, _, _ = _run(2, tmp_path, variant)
    one = serial[variant][0]
    assert one.shape == two.shape and numpy.abs(one[:, :2] - two[:, :2]).max() == 0
    assert numpy.abs(one[:, 2:5] - two[:, 2:5]).max() < 1e-9


@pytest.mark.parametrize("variant, nc", [("default", 3), ("gn", 6)])
def test_the_newton_jacobian_is_the_mass_matrix_of_every_unknown(serial, variant, nc):
    """getJacobian (the explicit scheme never calls it; an implicit time integration would): a symmetric matrix whose entries add up to
    nc times the water-plane area (the mass matrix of each of the nc unknowns, whose entries add up to the area)."""
    j = JACOBIANS[variant]
    assert j["nc"] == nc and j["finite"]
    assert j["total"] == pytest.approx(nc * j["area"], rel=1e-10)
    assert j["asymmetry"] < 1e-12
    assert j["nnz"] > j["n"]
