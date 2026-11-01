"""The immersed-boundary cylinder (RANS3PF, node partition, no overlap) on two ranks.

test/cylinder2D/ibm_method is the one test case that runs RANS3PF's immersed
boundary terms through a full multi-model solve. Its mesh is cut into two
subdomains with no overlap, so the faces between them are exterior faces of
each subdomain. Run with tight tolerances (the case's own are loose enough that
two runs of the same problem may stop at different iterates) it must give the
serial answer.

The case is copied and patched rather than edited: SuperLU is serial only, and
the tolerances are shrunk to 1e-10.
"""
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "cylinder2D", "ibm_method")


def _tool(name):
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    return candidate if os.path.exists(candidate) else shutil.which(name)


PARUN = _tool("parun")
MPIEXEC = _tool("mpiexec") or _tool("mpirun")


def _patched_copy(dest):
    shutil.copytree(CASE, dest, ignore=shutil.ignore_patterns("*.h5", "*.ymf", "*.log"))
    path = os.path.join(dest, "cylinder.py")
    with open(path) as f:
        text = f.read()
    text = text.replace("useSuperlu = True", "useSuperlu = False")
    text = re.sub(r"^([a-z_]*_nl_atol_res) = max\(1\.0e-10, [^)]*\)", r"\1 = 1.0e-10",
                  text, flags=re.M)
    with open(path, "w") as f:
        f.write(text)
    for name in os.listdir(dest):
        if name.endswith("_n.py"):
            n_path = os.path.join(dest, name)
            with open(n_path) as f:
                n_text = f.read()
            with open(n_path, "w") as f:
                f.write(n_text.replace("linTolFac = 0.01", "linTolFac = 1.0e-8"))


def _run(ranks, cwd):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    argv = [PARUN, "-l1", "cylinder_so.py", "-C", "T=0.3 onlySaveFinalSolution=True"]
    if MPIEXEC:  # even for one rank (test_archive_parallel_metadata.run_ranks says why)
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]


def _fields(cwd):
    """Velocity and pressure at the final time, ordered by node coordinates."""
    with h5py.File(os.path.join(cwd, "cylinder.h5"), "r") as h:
        result = {}
        for nodes, names in (("nodes_c0p2_Lagrange2", ("u_t2", "v_t2")),
                             ("nodesSpatial_Domain2", ("p_t2",))):
            x = numpy.array(h[nodes])[:, :2]
            order = numpy.lexsort((numpy.round(x[:, 1], 8), numpy.round(x[:, 0], 8)))
            result[nodes] = x[order]
            for name in names:
                result[name] = numpy.array(h[name]).ravel()[order]
        return result


@pytest.mark.skipif(PARUN is None or MPIEXEC is None, reason="parun or mpiexec not available")
def test_the_immersed_boundary_cylinder_on_two_ranks_gives_the_serial_answer(tmp_path):
    serial = str(tmp_path / "serial")
    parallel = str(tmp_path / "parallel")
    _patched_copy(serial)
    _patched_copy(parallel)
    _run(1, serial)
    _run(2, parallel)
    a, b = _fields(serial), _fields(parallel)
    numpy.testing.assert_allclose(a["nodes_c0p2_Lagrange2"], b["nodes_c0p2_Lagrange2"])
    for name in ("u_t2", "v_t2", "p_t2"):
        scale = numpy.abs(a[name]).max()
        assert scale > 1e-3, (name, scale)             # the flow has developed
        numpy.testing.assert_allclose(b[name], a[name], atol=1e-6 * scale, err_msg=name)
