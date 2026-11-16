"""
The H(div) spaces RT0 and BDM1 as FemTools spaces (`HdivOnSimplex`, `FaceDOFMap`, `Hdiv_AffineOnSimplex`):
normal continuity, sign and ordering rules, serial and 2 ranks (C8 round 5).

A dof is a flux density through a face along the face's global normal. Two elements, and two ranks, must
choose the same normal and, for BDM1, the same order of the nd dofs of the face; the map takes both from the
face's vertices sorted by *global* vertex number. `hdiv_dofmap_mpi.py` checks on each rank's subdomain (ghost
elements included), for both families on intervals, triangles and tetrahedra, with the local vertex orders
as generated and scrambled independently on each rank:

* for generic dof values the normal component is continuous across every face (round-off) while the
  tangential component jumps (the space is H(div), not H1);
* without the sign and ordering rule the normal component jumps (negative control);
* a field of the space is interpolated exactly, with its divergence, and the element fluxes add up to
  |K| div v.

This file adds what needs all ranks together: the global dof count, the owned blocks tiling the global range,
ranks that share a dof agreeing on its point, its face and its VALUE (so on the direction of the normal), the
dof table being the serial one, and the interpolation orders (RT0 first, BDM1 second in L2, both first in the
divergence).
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
HELPER = os.path.join(HERE, "hdiv_dofmap_mpi.py")
FAMILIES = ("rt0", "bdm1")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, shape, part="node", layers=1, rates=False):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update(SHAPE=shape, PART_TYPE=part, NLAYERS=str(layers), OUT=str(cwd), RATES="1" if rates else "0")
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    rate_rows = [json.loads(line[len("RATES "):]) for line in out.stdout.splitlines() if line.startswith("RATES ")]
    assert results, out.stdout[-2000:] + out.stderr[-2000:]
    return results, rate_rows


def _case(r):
    return (r["family"], r["scrambled"])


def _check_rank_results(results, shape):
    """What every rank must satisfy on its own subdomain."""
    for r in results:
        tag = "rank %d %s %s scrambled=%s" % (r["rank"], shape, r["family"], r["scrambled"])
        assert r["nAll"] == r["nAll_expected"], tag
        assert r["s2g_injective"] and r["owned_first"] and r["ghosts_not_owned"] and r["l2g_dtype_ok"], tag
        assert r["unused_dofs"] == 0, tag
        assert r["defect"] < 1.0e-13, tag
        assert r["nFaces2"] > 0, tag
        assert r["normal_jump"] < 1.0e-11, "%s: normal jump %g" % (tag, r["normal_jump"])
        assert r["normal_jump_unoriented"] > 1.0e-1, tag           # negative control
        if shape != "edge":                                          # in 1D there is no tangential direction
            assert r["tangential_jump"] > 1.0e-1, tag
        assert r["negative_signs"] > 0, tag
        assert r["field_dof_defect"] < 1.0e-13 and r["field_error"] < 1.0e-12, tag
        if not r["scrambled"]:
            assert r["space_class"] == "Hdiv_AffineOnSimplex_" + r["family"], tag
            assert r["space_nDOF"] == r["nDOF"] == r["function_dofs"], tag
            assert r["all_known"] and r["same_as_bare_map"], tag
            assert r["interpolation_defect"] < 1.0e-13 and r["interpolation_error"] < 1.0e-12, tag
            assert r["divergence_error"] < 1.0e-12 and r["flux_balance"] < 1.0e-13, tag


def _global_table(cwd, shape, case, ranks):
    """(points, values) per global dof, after checking that the ranks holding a dof agree on both."""
    family, scrambled = case
    tables = [np.load(os.path.join(str(cwd), "%s_%s_%d_r%d.npz" % (shape, family, int(scrambled), r)))
              for r in range(ranks)]
    n = int(max(t["s2g"].max() for t in tables)) + 1
    x = np.full((n, 6), np.nan)
    v = np.full((n,), np.nan)
    owned = 0
    for t in tables:
        s2g = t["s2g"]
        seen = ~np.isnan(v[s2g])
        assert np.abs(x[s2g[seen]] - t["x"][seen]).max(initial=0.0) < 1.0e-13, "ranks disagree on the point of a shared dof"
        assert np.abs(v[s2g[seen]] - t["dof"][seen]).max(initial=0.0) < 1.0e-13, \
            "ranks disagree on the value of a shared dof (on the face's normal) by %g" % np.abs(v[s2g[seen]] - t["dof"][seen]).max()
        x[s2g] = t["x"]
        v[s2g] = t["dof"]
        owned += int(t["nOwned"])
    assert not np.isnan(v).any(), "a global dof is on no rank"
    assert owned == n, "owned blocks do not tile the global dofs: %d != %d" % (owned, n)
    return x, v


def _sorted(x, v):
    order = np.lexsort(np.round(x, 9).T[::-1])
    return x[order], v[order]


@pytest.mark.parametrize("shape", ["edge", "tri", "tet"])
def test_serial_and_interpolation_orders(tmp_path, shape):
    results, rates = _run(1, tmp_path, shape, rates=True)
    _check_rank_results(results, shape)
    assert sorted({_case(r) for r in results}) == [(f, s) for f in sorted(FAMILIES) for s in (False, True)]
    assert sorted(r["family"] for r in rates) == sorted(FAMILIES)
    for r in rates:
        # in 1D RT0 and BDM1 are both the continuous piecewise linears
        expected = 2 if (r["family"] == "bdm1" or shape == "edge") else 1
        assert abs(r["L2_rate"] - expected) < 0.1, r
        assert abs(r["div_rate"] - 1.0) < 0.1, r


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("shape,part", [("edge", "node"), ("tri", "node"), ("tri", "element"), ("tet", "node"), ("tet", "element")])
def test_two_ranks_are_conforming_and_span_the_serial_space(tmp_path, shape, part):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    serial, _ = _run(1, tmp_path / "1", shape, part)
    parallel, _ = _run(2, tmp_path / "2", shape, part)
    _check_rank_results(parallel, shape)
    assert {r["rank"] for r in parallel} == {0, 1}
    if shape != "edge":
        # the partition exercises what the rule is for: on some rank the subdomain-local vertex numbers order a
        # face's vertices differently from the global ones (a normal or a dof order taken from local numbers
        # would differ between the ranks)
        assert max(r["element_faces_local_order_differs_from_global"] for r in parallel) > 0
    cases = sorted({_case(r) for r in serial})
    assert cases == sorted({_case(r) for r in parallel})
    for case in cases:
        nAll = {r["nAll"] for r in serial + parallel if _case(r) == case}
        assert len(nAll) == 1, "global dof count differs between serial and 2 ranks: %s %s" % (case, nAll)
        offsets = {tuple(r["offsets"]) for r in parallel if _case(r) == case}
        assert len(offsets) == 1 and list(offsets)[0][-1] == list(nAll)[0]
        xs, vs = _sorted(*_global_table(tmp_path / "1", shape, case, 1))
        xp, vp = _sorted(*_global_table(tmp_path / "2", shape, case, 2))
        assert xs.shape == xp.shape == (list(nAll)[0], 6)
        # the same dofs (point and face), and the same values: the numbering differs between serial and
        # parallel, the space and the direction of every normal do not
        assert np.abs(xs - xp).max() < 1.0e-12, case
        assert np.abs(vs - vp).max() < 1.0e-12, case
        # one dof per (point, face)
        d = np.diff(xs, axis=0)
        assert np.abs(d).max(axis=1).min() > 1.0e-6, case
