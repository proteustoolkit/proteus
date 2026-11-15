"""
The any-order quadrilateral/hexahedral dof map (`FemTools.CubeEntityDOFMap`) and space
(`C0_AffinePolynomialOnCube`): conformity across shared edges and faces in serial and on 2 ranks (C8 round 4).

For p >= 3 an edge carries more than one dof and a quadrilateral face (p-1)^2, so two elements that list the
shared vertices in different local orders must still identify the same points. The map indexes an edge's
dofs from its vertex with the smaller *global* number, and a face's dofs in the frame whose origin is the
vertex with the smallest global number and whose first axis points to the neighbour with the smaller one.
`cube_dofmap_mpi.py` checks, on each rank's subdomain (ghost elements included), for GLL and equispaced
nodes, p = 1..5 on quadrilaterals and 1..4 on hexahedra, with the local vertex orders as generated and with
every element listed through a random symmetry of the square (8) or cube (48), independently on each rank:

* the function with generic dof values is continuous across every face (jump at round-off);
* without the orientation rule it is not (the negative control: the test would see a wrong map);
* p = 1 reproduces `NodalDOFMap` exactly; p = 2 has the dof points of `QuadraticLagrangeCubeDOFMap`, and
  on hexahedra its numbering too (local-to-global and `subdomain2global`, serial and 2 ranks), so the
  closed-form parallel numbering equals the one partitioning.cpp gathers;
* a tensor polynomial of degree p is interpolated exactly through the space class.

This file adds what needs all ranks together: the global dof count, the owned blocks tiling the global range,
ranks that share a dof agreeing on its point, and the set of dof points being the serial one.
"""
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "cube_dofmap_mpi.py")
PMAX = {"quad": 5, "hex": 4}


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, shape, part="node", layers=1):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update(SHAPE=shape, PART_TYPE=part, NLAYERS=str(layers), OUT=str(cwd))
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert results, out.stdout[-2000:] + out.stderr[-2000:]
    return results


def _case(r):
    return (r["p"], r["nodes"], r["scrambled"])


def _check_rank_results(results, shape):
    """What every rank must satisfy on its own subdomain."""
    for r in results:
        tag = "rank %d %s p=%d %s scrambled=%s" % (r["rank"], shape, r["p"], r["nodes"], r["scrambled"])
        assert r["nAll"] == r["nAll_expected"], tag
        assert r["s2g_injective"] and r["owned_first"] and r["ghosts_not_owned"], tag
        assert r["l2g_dtype_ok"], tag
        # every dof is used by an element, up to the dofs of edge records that are not edges (see the helper)
        assert r["unused_dofs"] == (r["p"] - 1) * r["junk_edges"], tag
        assert r["defect"] < 1.0e-13, tag
        assert r["jump"] < 1.0e-12, "%s: jump %g across a face" % (tag, r["jump"])
        assert r["nFaces2"] > 0, tag
        if r["p"] >= 3:
            # negative control: without the rule the map is not conforming, on the generated mesh already
            assert r["defect_unoriented"] > 1.0e-3 and r["jump_unoriented"] > 1.0e-3, tag
        if not r["scrambled"]:
            assert r["interpolation_error"] < 1.0e-12, "%s: interpolation error %g" % (tag, r["interpolation_error"])
            assert r["space_class"] == "C0_AffinePolynomialOnCube_q%d_%s" % (r["p"], r["nodes"]), tag
            assert r["space_nDOF"] == r["nDOF"], tag
            if r["p"] == 1:
                assert r["equals_existing_map"], tag
            if r["p"] == 2:
                assert r["q2_same_points"], tag
                assert r["q2_same_numbering"] == (shape == "hex"), tag


def _global_points(cwd, shape, case, ranks):
    """One point per global dof, after checking that the ranks holding a dof agree on its point."""
    p, nodes, scrambled = case
    tables = [np.load(os.path.join(str(cwd), "%s_p%d_%s_%d_r%d.npz" % (shape, p, nodes, int(scrambled), r)))
              for r in range(ranks)]
    n = int(max(t["s2g"].max() for t in tables)) + 1
    x = np.full((n, 3), np.nan)
    owned = 0
    for t in tables:
        s2g, pts = t["s2g"], t["x"]
        seen = ~np.isnan(x[s2g, 0])
        assert np.abs(x[s2g[seen]] - pts[seen]).max(initial=0.0) < 1.0e-13, \
            "ranks disagree on the point of a shared dof by %g" % np.abs(x[s2g[seen]] - pts[seen]).max()
        x[s2g] = pts
        owned += int(t["nOwned"])
        assert int(t["usedOwned"]) == int(t["nOwned"]), "an owned dof belongs to no element of its rank"
    assert not np.isnan(x).any(), "a global dof is on no rank"
    assert owned == n, "owned blocks do not tile the global dofs: %d != %d" % (owned, n)
    return x


def _sorted(x):
    return x[np.lexsort(np.round(x, 9).T[::-1])]


@pytest.mark.parametrize("shape", ["quad", "hex"])
def test_serial(tmp_path, shape):
    results = _run(1, tmp_path, shape)
    _check_rank_results(results, shape)
    assert {r["p"] for r in results} == set(range(1, PMAX[shape] + 1))
    assert all(r["junk_edges"] == 0 for r in results)
    if shape == "hex":
        # the scrambled listings show the map every one of the eight frames a face can have
        assert max(r["face_frames_seen"] for r in results if r["scrambled"]) == 8


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("shape,part", [("quad", "node"), ("quad", "element"), ("hex", "node"), ("hex", "element")])
def test_two_ranks_are_conforming_and_span_the_serial_space(tmp_path, shape, part):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    serial = _run(1, tmp_path / "1", shape, part)
    parallel = _run(2, tmp_path / "2", shape, part)
    _check_rank_results(parallel, shape)
    assert {r["rank"] for r in parallel} == {0, 1}
    # the partition exercises what the rule is for: on some rank the subdomain-local vertex numbers order an
    # edge differently from the global ones (orienting by local numbers would differ between the ranks)
    assert max(r["edges_local_order_differs_from_global"] for r in parallel) > 0
    assert max(r["element_edges_reversed"] for r in parallel) > 0
    if shape == "hex":
        assert max(r["element_faces_reoriented"] for r in parallel) > 0
    cases = sorted({_case(r) for r in serial})
    assert cases == sorted({_case(r) for r in parallel})
    for case in cases:
        nAll = {r["nAll"] for r in serial + parallel if _case(r) == case}
        assert len(nAll) == 1, "global dof count differs between serial and 2 ranks: %s %s" % (case, nAll)
        offsets = {tuple(r["offsets"]) for r in parallel if _case(r) == case}
        assert len(offsets) == 1 and list(offsets)[0][-1] == list(nAll)[0]
        xs = _global_points(tmp_path / "1", shape, case, 1)
        xp = _global_points(tmp_path / "2", shape, case, 2)
        assert xs.shape == xp.shape == (list(nAll)[0], 3)
        # the same set of dof points (the numbering differs between serial and parallel, the space does not)
        assert np.abs(_sorted(xs) - _sorted(xp)).max() < 1.0e-12, case
        # and the points are distinct: one dof per point
        d = np.diff(_sorted(xs), axis=0)
        assert np.abs(d).max(axis=1).min() > 1.0e-6, case
