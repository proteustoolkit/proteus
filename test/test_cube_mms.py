"""
Poisson with a manufactured solution on the any-order Q_p spaces on quadrilaterals and hexahedra, through
NS_base and OneLevelTransport (C8 round 4): the first model that runs on `C0_AffinePolynomialOnCube`. The
counterpart of test_simplex_mms.py.

`cube_mms_mpi.py` solves -div(grad u) = f on the unit square or cube with strong Dirichlet data on the whole
boundary (data that is neither zero nor linear on any face) and integrates the errors itself.

* Convergence: L2 rate p+1 and H1 rate p, within 0.15, for p = 1..5 on quadrilaterals and p = 1..4 on
  hexahedra (GLL nodes), on the pair of meshes listed per degree.
* A solution that is a polynomial of degree p in each coordinate (a member of Q_p that is not in P_p) is
  reproduced to round-off.
* Equispaced nodes give the solution of the same space with another boundary interpolant.
* 2 ranks give the serial errors to solver tolerance, node and element partitions.
* The Jacobian's sparsity is exactly the element couplings of the dof map, on every rank.
* The archive holds the complete dof vector and the sub-cell view of the space (order**nd quadrilaterals or
  hexahedra per element, none inverted, total volume 1), and each (order, nodes) is its own class.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "cube_mms_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(tmp_path, ranks=1, **case):
    cwd = tmp_path / ("_".join("%s%s" % kv for kv in sorted(case.items())) + "_r%d" % ranks)
    cwd.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update({k: str(v) for k, v in case.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, out.stdout[-2000:] + out.stderr[-2000:]
    r = results[0]
    # what every run must satisfy
    assert r["affine_defect"] < 1.0e-13, r      # the helper's own error integration assumes affine elements
    assert r["nnz_equal_on_every_rank"] and r["nnz_model"] == r["nnz_brute"], r
    # the dof vector is read back exactly; a ghost dof's point is recomputed on this rank (round-off)
    assert r["archive_u"] == 0.0 and r["archive_x"] < 1.0e-13, r
    assert r["archive_nDOF"] == r["nDOF"], r
    assert r["archive_nSub"] == r["p"] ** r["nd"] and r["archive_cells"] == r["nElements"] * r["archive_nSub"], r
    assert r["archive_vertices"] == 2 ** r["nd"], r
    assert abs(r["archive_volume"] - 1.0) < 1.0e-12 and r["archive_min_cell"] > 0.0, r
    assert r["space_class"] == "C0_AffinePolynomialOnCube_q%d_%s" % (r["p"], r["nodes"]), r
    assert r["nDOF"] == (r["p"] * (r["nn"] - 1) + 1) ** r["nd"], r
    return r


# (nd, p): the two meshes (nodes per direction) the rate is measured on (measured 2026-11-14: L2 rates
# 2.00 2.98 3.99 4.94 6.01 in 2D and 2.00 2.90 3.98 4.90 in 3D; H1 1.00 1.99 3.01 3.95 5.01 and 1.02 1.96 3.02 3.92)
RATE_MESHES = {(2, 1): (9, 17), (2, 2): (9, 17), (2, 3): (5, 9), (2, 4): (5, 9), (2, 5): (5, 9),
               (3, 1): (9, 17), (3, 2): (5, 9), (3, 3): (5, 7), (3, 4): (5, 6)}


@pytest.mark.parametrize("nd,p", sorted(RATE_MESHES))
def test_convergence_rates(tmp_path, nd, p):
    n1, n2 = RATE_MESHES[(nd, p)]
    coarse = _run(tmp_path, ND=nd, P=p, NN=n1)
    fine = _run(tmp_path, ND=nd, P=p, NN=n2)
    ratio = math.log((n2 - 1.0) / (n1 - 1.0))
    l2 = math.log(coarse["L2"] / fine["L2"]) / ratio
    h1 = math.log(coarse["H1"] / fine["H1"]) / ratio
    assert abs(l2 - (p + 1)) < 0.15, "L2 rate %.2f, expected %d (%g -> %g)" % (l2, p + 1, coarse["L2"], fine["L2"])
    assert abs(h1 - p) < 0.15, "H1 rate %.2f, expected %d (%g -> %g)" % (h1, p, coarse["H1"], fine["H1"])


@pytest.mark.parametrize("nd,p,nodes", [(2, 1, "gll"), (2, 3, "gll"), (2, 5, "gll"), (2, 3, "equispaced"),
                                        (3, 2, "gll"), (3, 4, "gll"), (3, 3, "equispaced")])
def test_a_tensor_polynomial_of_degree_p_is_reproduced(tmp_path, nd, p, nodes):
    r = _run(tmp_path, ND=nd, P=p, NN=4 if nd == 2 else 3, SOLUTION="poly", NODES=nodes)
    assert r["L2"] < 1.0e-11 and r["H1"] < 1.0e-10 and r["Linf"] < 1.0e-11, r


@pytest.mark.parametrize("nd,p,nn", [(2, 4, 9), (3, 3, 5)])
def test_equispaced_nodes_give_the_solution_of_the_same_space(tmp_path, nd, p, nn):
    gll = _run(tmp_path, ND=nd, P=p, NN=nn)
    equispaced = _run(tmp_path, ND=nd, P=p, NN=nn, NODES="equispaced")
    assert equispaced["nDOF"] == gll["nDOF"]
    # the same space, another boundary interpolant (measured: 5.6% in L2 for Q3 on 4^3 hexahedra)
    assert abs(equispaced["L2"] - gll["L2"]) < 0.10 * gll["L2"]
    assert abs(equispaced["L2"] - gll["L2"]) > 1.0e-9 * gll["L2"]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("nd,p,nn,part,layers", [(2, 3, 9, "node", 1),
                                                 (2, 3, 9, "element", 0),
                                                 (2, 5, 5, "element", 1),
                                                 (3, 3, 5, "node", 1),
                                                 (3, 4, 4, "element", 0),
                                                 (3, 3, 4, "element", 1)])
def test_two_ranks_give_the_serial_solution(tmp_path, nd, p, nn, part, layers):
    serial = _run(tmp_path, ND=nd, P=p, NN=nn, PART_TYPE=part, NLAYERS=layers)
    parallel = _run(tmp_path, ranks=2, ND=nd, P=p, NN=nn, PART_TYPE=part, NLAYERS=layers)
    assert parallel["ranks"] == 2 and parallel["nDOF"] == serial["nDOF"] and parallel["nElements"] == serial["nElements"]
    for norm in ("L2", "H1", "Linf"):
        assert abs(parallel[norm] - serial[norm]) < 1.0e-8 * serial[norm], (norm, serial[norm], parallel[norm])


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_more_ranks_than_cells_stops_at_once_instead_of_hanging(tmp_path):
    """NN=2 is one cell per direction: on 2 ranks one rank has no element. Before the guard that rank died in
    Transport.initializeJacobian (nNonzerosInJacobian > 0) and the other waited for it past 300 s."""
    cwd = tmp_path / "toofewcells"
    cwd.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", ND="3", P="3", NN="2", PART_TYPE="element", NLAYERS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    proc = subprocess.run([MPIEXEC, "-n", "2", sys.executable, HELPER], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode != 0
    assert "every rank needs a cell" in proc.stdout + proc.stderr
