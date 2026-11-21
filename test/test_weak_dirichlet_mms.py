"""Weak (Nitsche) Dirichlet data on the any-order simplex spaces (C0 round 12).

`weak_dirichlet_mms_mpi.py` solves the problem of `simplex_mms_mpi.py` (-div(grad u) = f, smooth u that is neither
zero nor linear on any face) imposing u = g on the whole boundary weakly, with `Diffusion_{SIPG,IIPG}_exterior`, and
integrates the errors itself.

The Nitsche penalty is penalty_constant/h; coercivity needs it to grow like p**2/h. `NumericalFlux.penaltyOrderFactor`
supplies p**2 above degree 2 (degrees 1 and 2 keep the constant they always had). Measured 2026-11-16 (macOS arm64,
penalty_constant 10, h = 1/16): without the factor the L2 error is 3.6x the strong-Dirichlet error at p = 3 and 490x at
p = 4 (and it grows when the mesh is refined); with it the weak and strong errors agree to 2% and the rates are those of the
strong problem.

* the rate on the last two meshes, L2 p+1 and H1 p within 0.15 (p = 3, 4);
* the weak error is within 10% of the strong-Dirichlet error on the same mesh (SIPG, IIPG, NIPG; Lagrange = Bernstein);
* the negative control: with the factor off, the error is at least 3x the strong one;
* 2 ranks give the serial errors; the same on tetrahedra (p = 3);
* the factor itself.
"""
import importlib.util
import json
import math
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(tmp_path_factory, helper, ranks=1, **case):
    cwd = tmp_path_factory.mktemp("_".join("%s%s" % kv for kv in sorted(case.items())) + "_r%d" % ranks)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update({k: str(v) for k, v in case.items()})
    script = os.path.join(HERE, helper)
    argv = [sys.executable, script] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, script]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, out.stdout[-2000:] + out.stderr[-2000:]
    return results[0]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cache = {}

    def go(weak=True, ranks=1, **case):
        key = (weak, ranks, tuple(sorted(case.items())))
        if key not in cache:
            helper = "weak_dirichlet_mms_mpi.py" if weak else "simplex_mms_mpi.py"
            if weak:
                if case.pop("penalty", True) and "PENALTY" not in case:
                    case = dict(case, PENALTY=10)
            cache[key] = _run(tmp_path_factory, helper, ranks, **case)
        return cache[key]
    return go


def _rate(a, b, ha, hb):
    return math.log(a / b) / math.log(ha / hb)


@pytest.mark.parametrize("p", [3, 4])
def test_weak_dirichlet_converges_at_the_optimal_rate_and_matches_strong(run, p):
    coarse, fine = run(ND=2, P=p, NN=9), run(ND=2, P=p, NN=17)
    assert _rate(coarse["L2"], fine["L2"], coarse["h"], fine["h"]) > p + 1 - 0.15, (coarse, fine)
    assert _rate(coarse["H1"], fine["H1"], coarse["h"], fine["h"]) > p - 0.15, (coarse, fine)
    strong = run(weak=False, ND=2, P=p, NN=17)
    assert fine["L2"] < 1.1 * strong["L2"] and fine["H1"] < 1.1 * strong["H1"], (fine, strong)


@pytest.mark.parametrize("flux", ["iipg", "nipg"])
def test_the_other_boundary_forms_match_strong(run, flux):
    r, strong = run(ND=2, P=3, NN=9, FLUX=flux), run(weak=False, ND=2, P=3, NN=9)
    assert r["L2"] < 1.1 * strong["L2"] and r["H1"] < 1.1 * strong["H1"], (r, strong)


def test_bernstein_is_the_same_space(run):
    r, lagrange = run(ND=2, P=3, NN=9, FAMILY="bernstein"), run(ND=2, P=3, NN=9)
    assert abs(r["L2"] - lagrange["L2"]) < 1.0e-6 * lagrange["L2"] and r["family"] == "bernstein"


@pytest.mark.parametrize("p,factor", [(3, 3.0), (4, 40.0)])
def test_without_the_degree_factor_the_error_is_much_larger(run, p, factor):
    off = run(ND=2, P=p, NN=17, SCALING=0)
    strong = run(weak=False, ND=2, P=p, NN=17)
    assert off["L2"] > factor * strong["L2"], (off, strong)


@pytest.mark.parametrize("nd,p,nn", [(2, 3, 9), (2, 4, 5), (3, 3, 5)])
def test_two_ranks_give_the_serial_errors(run, nd, p, nn):
    s, r = run(ND=nd, P=p, NN=nn), run(ranks=2, ND=nd, P=p, NN=nn)
    assert r["ranks"] == 2
    assert abs(r["L2"] - s["L2"]) < 1.0e-6 * s["L2"] and abs(r["H1"] - s["H1"]) < 1.0e-6 * s["H1"], (s, r)


def test_tetrahedra_p3_match_strong_and_need_the_factor(run):
    r, strong, off = run(ND=3, P=3, NN=5), run(weak=False, ND=3, P=3, NN=5), run(ND=3, P=3, NN=5, SCALING=0)
    assert r["L2"] < 1.1 * strong["L2"] and r["H1"] < 1.1 * strong["H1"], (r, strong)
    assert off["L2"] > 10 * strong["L2"], (off, strong)


@pytest.mark.parametrize("p", [3, 4])
def test_dg_interior_penalty_converges_at_the_optimal_rate(run, p):
    """Discontinuous monomial P3/P4, SIPG interior and boundary faces (the class's own constant, 100)."""
    coarse, fine = run(DG=1, FLUX="dgsipg", ND=2, P=p, NN=9, penalty=False), run(DG=1, FLUX="dgsipg", ND=2, P=p, NN=17, penalty=False)
    assert _rate(coarse["L2"], fine["L2"], coarse["h"], fine["h"]) > p + 1 - 0.3, (coarse, fine)
    assert _rate(coarse["H1"], fine["H1"], coarse["h"], fine["h"]) > p - 0.3, (coarse, fine)


@pytest.mark.parametrize("p,factor", [(4, 10.0), (5, 5.0)])
def test_dg_penalty_floor_holds_the_error_where_a_small_constant_loses_it(run, p, factor):
    """With the exterior classes' constant 10 the SIPG error at degree 4, 5 is 36x, 7.8x that of a large penalty (1000)
    on the coarse mesh (h = 1/8); the floor 2.5 p**2 of NumericalFlux.penaltyOrderFactor brings it to the large-penalty value."""
    floor = run(DG=1, FLUX="dgsipg", ND=2, P=p, NN=9)
    off = run(DG=1, FLUX="dgsipg", ND=2, P=p, NN=9, SCALING=0)
    big = run(DG=1, FLUX="dgsipg", ND=2, P=p, NN=9, PENALTY=1000)
    assert floor["L2"] < 1.2 * big["L2"] and floor["H1"] < 1.2 * big["H1"], (floor, big)
    assert off["L2"] > factor * big["L2"], (off, big)


@pytest.mark.parametrize("p", [3, 4])
def test_dg_two_ranks_give_the_serial_errors(run, p):
    """The floor is applied on every rank from the space, so a partitioned run is the serial one (element partition,
    one layer of overlap, as the discontinuous coupling requires)."""
    case = dict(DG=1, FLUX="dgsipg", ND=2, P=p, NN=9, PART_TYPE="element", NLAYERS=1)
    s, r = run(**case), run(ranks=2, **case)
    assert r["ranks"] == 2
    assert abs(r["L2"] - s["L2"]) < 1.0e-6 * s["L2"] and abs(r["H1"] - s["H1"]) < 1.0e-6 * s["H1"], (s, r)


def test_the_degree_factor():
    from types import SimpleNamespace
    from proteus import Comm
    Comm.init()
    from proteus import FemTools, MeshTools, NumericalFlux
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(3, 3, 1.0, 1.0)
    mesh.partitionMesh(nLayersOfOverlap=0, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
    mesh = mesh.subdomainMesh

    def factor(*spaces, scaling=True, constant=10.0):
        vt = SimpleNamespace(u={i: SimpleNamespace(femSpace=s) for i, s in enumerate(spaces)})
        nf = SimpleNamespace(vt=vt, penalty_order_scaling=scaling, penalty_constant=constant)
        return NumericalFlux.NF_base.penaltyOrderFactor(nf)
    P = lambda p, f="lagrange": FemTools.C0_AffinePolynomialOnSimplex(mesh, 2, p, f)
    assert [factor(P(p)) for p in (1, 2, 3, 4, 5)] == [1.0, 1.0, 9.0, 16.0, 25.0]
    assert factor(P(3, "bernstein")) == 9.0
    assert factor(FemTools.C0_AffineLinearOnSimplexWithNodalBasis(mesh, 2), FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis(mesh, 2)) == 1.0
    assert factor(P(2), P(4)) == 16.0            # the largest degree among the unknowns
    assert factor(P(4), scaling=False) == 1.0
    # discontinuous spaces: a floor of 2.5 p**2 on the penalty constant, not a factor
    DG = lambda p: getattr(FemTools, "DG_AffineP%d_OnSimplexWithMonomialBasis" % p)(mesh, 2)
    assert [factor(DG(p)) for p in (1, 2, 3, 4, 5)] == [1.0, 1.0, 2.25, 4.0, 6.25]      # constant 10
    assert [factor(DG(p), constant=100.0) for p in (3, 4, 5, 6)] == [1.0, 1.0, 1.0, 1.0]   # the interior classes' constant
    assert factor(DG(4), scaling=False) == 1.0
    assert factor(FemTools.DG_AffineQuadraticOnSimplexWithNodalBasis(mesh, 2)) == 1.0
