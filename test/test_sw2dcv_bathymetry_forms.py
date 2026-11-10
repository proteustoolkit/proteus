"""The bathymetry reaches b.dof the same way in every accepted form, on 1 rank and on 2.

SW2DCV and GN_SW2DCV take the bathymetry as a function of the coordinates, or as an array in the input mesh's node
order: shape (N,), or (N, k) with k = 1, 2, 3 whose last column is the bathymetry (the columns before it the node's
coordinates). On more than one rank the coefficients took the column of the 1- and 3-column forms (the 2-column form
raised UnboundLocalError) and permuted it to the subdomain with the mesh's node map; on one rank they assigned the
array itself, so every 2-D form raised "could not broadcast input array from shape (149,k) into shape (149,)". One
rule (SW2DCV.bathymetryAtNodes) now serves both. obstacle_flow on its stored Triangle mesh (149 nodes); the array forms carry
b = 0.01 x + 0.02 y + 0.001 i (i the file node), the function b = 0.01 x + 0.02 y; every node of every subdomain,
ghosts included, is checked against that analytic value, in b.dof and in mesh.nodeArray[:, 2].
"""
import importlib.util
import os
import shutil
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
# a failed launch keeps its whole output (loaded by path: --import-mode=importlib)
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "sw2dcv_bathymetry_forms_mpi.py")
NNODES = 149
FORMS = ["callable", "1d", "col1", "col2", "col3"]


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _bathymetry(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    launch_output.run([MPIEXEC, "-n", str(ranks), sys.executable, HELPER], cwd=cwd, env=env, timeout=600)
    shapes = {open(os.path.join(str(cwd), "bshape_rank%d.txt" % r)).read() for r in range(ranks)}
    return shapes, [numpy.loadtxt(os.path.join(str(cwd), "b_rank%d.csv" % r), delimiter=",", skiprows=1)
                    for r in range(ranks)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("model", ["SW2DCV", "GN_SW2DCV"])
@pytest.mark.parametrize("form", FORMS)
def test_bathymetry_form_serial_and_two_ranks(tmp_path, model, form):
    values = {}
    for ranks in (1, 2):
        d = tmp_path / ("np%d" % ranks)
        d.mkdir()
        overrides = {"FORM": form}
        if model == "GN_SW2DCV":
            overrides["GN"] = "1"
        shapes, arrays = _bathymetry(ranks, d, **overrides)
        assert all(s.startswith("(") and s.count(",") == 1 and s.endswith(",)") for s in shapes), \
            (ranks, "b.dof is not one value per node", shapes)
        seen = set()
        for a in arrays:
            x, y, i, b, z, expected = a.T
            err = numpy.abs(b - expected).max()
            assert err < 1e-12, (ranks, "b.dof differs from the analytic bathymetry by %.3e" % err)
            assert numpy.array_equal(z, b), (ranks, "mesh.nodeArray[:, 2] is not b.dof")
            seen.update(i.astype(int).tolist())
            for ii, bb in zip(i.astype(int), b):
                values.setdefault(ii, set()).add(bb)
        assert seen == set(range(NNODES)), ranks
    # serial and 2 ranks, every copy of a node: one value
    assert all(len(v) == 1 for v in values.values())
