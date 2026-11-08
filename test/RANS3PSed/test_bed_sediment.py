"""RANS3PSed (three-phase: sediment, water, air) on air-water-vv's bed_sediment deck, serial against 2 and 4 ranks.

The first RANS3PSed deck in proteus's tests (bed_sediment/NOTICE.md: copied from erdc/air-water-vv). 0.02 s at
refinement 10 (a 15 x 30 cell domain, sediment column below the water): the sediment velocity reaches 0.18 m/s.
Every model solves with a direct solver and the partition is pinned ('current'), so the ranks must reproduce the
serial fields at the archived steps to round-off. Measured 2026-11-08 (macOS arm64, MPICH): worst relative difference
over vos, u, v, us, vs, p, pInc, vof, phi 1.4e-13 on 2 ranks, 8.2e-14 on 4.
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "bed_sediment")
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "..", "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
PREFIXES = ("vos_", "vof_", "ncls_", "rdls_", "mcorr_", "rans3p_sed_", "rans2p_", "phi_", "pressure_", "pinit_")
FIELDS = ("vos", "u", "v", "us", "vs", "p", "pInc", "vof", "phi")
CONTEXT = "duration=0.02 dtout=0.01 refinement=10"


def _bin(name):
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    return candidate if os.path.exists(candidate) else shutil.which(name)


MPIEXEC = _bin("mpiexec") or _bin("mpirun")
PARUN = _bin("parun")


def _direct_solver():
    from petsc4py import PETSc
    for package in ("superlu_dist", "mumps"):
        if PETSc.Sys.hasExternalPackage(package):
            return package
    return None


def _run(tmp_path, ranks):
    cwd = tmp_path / ("r%d" % ranks)
    shutil.copytree(DECK, cwd)        # staged by the test process, before any rank starts
    with open(cwd / "petsc.options.lu", "w") as f:
        for prefix in PREFIXES:
            f.write("-%sksp_type preonly -%spc_type lu -%spc_factor_mat_solver_type %s\n"
                    % (prefix, prefix, prefix, _direct_solver()))
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    argv = [MPIEXEC, "-n", str(ranks), sys.executable, PARUN, "-l", "1", "tank_so.py",
            "-O", "petsc.options.lu", "-C", CONTEXT]
    launch_output.run(argv, cwd, env, timeout=900)
    return cwd / "tank.h5"


def _fields(path):
    import h5py
    with h5py.File(path, "r") as f:
        step = max(int(k[len("nodesSpatial_Domain"):]) for k in f.keys() if k.startswith("nodesSpatial_Domain"))
        X = numpy.array(f["nodesSpatial_Domain%d" % step])
        order = numpy.lexsort(numpy.round(X[:, :2], 9).T[::-1])
        return step, X[order], {name: numpy.array(f["%s_t%d" % (name, step)])[order] for name in FIELDS}


@pytest.mark.skipif(MPIEXEC is None or PARUN is None, reason="needs mpiexec and parun")
@pytest.mark.parametrize("ranks", [2, 4])
def test_bed_sediment_on_several_ranks_matches_serial(ranks, tmp_path):
    if _direct_solver() is None:
        pytest.skip("needs superlu_dist or mumps")
    step, X1, serial = _fields(_run(tmp_path, 1))
    stepN, XN, parallel = _fields(_run(tmp_path, ranks))
    assert step == stepN and step >= 2, (step, stepN)
    numpy.testing.assert_allclose(XN, X1, atol=1e-12)
    # the sediment phase moves: the run exercises RANS3PSed, not a resting column
    assert numpy.abs(serial["vs"]).max() > 0.05, numpy.abs(serial["vs"]).max()
    worst = {name: numpy.abs(parallel[name] - serial[name]).max() / max(numpy.abs(serial[name]).max(), 1e-300)
             for name in FIELDS}
    assert max(worst.values()) < 1e-10, worst
