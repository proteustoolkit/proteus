"""Per-face strong | weak | none boundary kinds (C12), measured on a manufactured advection-diffusion problem.

`mixed_bc_mms_mpi.py` solves  -eps lap(u) + b.grad(u) = f  on the unit square with a different kind on each wall:
strong on the left (the rows of its dofs are replaced), weak on the bottom and top (Nitsche, upwind inflow value) and a
declared open face ('none': no value, the upwinded outflow and a zero diffusive flux) on the right, where the exact normal
derivative vanishes. The exact solution is the data of every strong and weak wall.

* optimal rates on the mixed problem (P2: L2 3, H1 2; P3: L2 4, H1 3);
* a strong face gets no weak term (no face point is both), and the dofs of the strong wall hold the data to round-off;
* a Dirichlet function that names no kind under a weak flux class stays weak: the same run as an undeclared setter, bit for bit
  (the default decided 2026-11-17: no existing result moves);
* serial = 2 and 3 ranks, node and element partitions;
* a face declared 'none' under the ``noFlow`` component of StrongDirichlet takes the upwinded outflow on that face only: the same
  run as the component's ``outFlow`` flag with the face undeclared (the old way), and the wall (nothing declared) is 100x worse;
* a face declared 'weak' under a class that has no weak term is refused, not ignored;
* TADR (its kernel reads the kind per face): a boundary layer with the left wall strong and the right weak equals the all-weak
  run, with the left weak and the right strong the all-strong run (the legacy ``forceStrongConditions``), serial = 2 ranks.
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
MIXED = "left=strong,bottom=weak,top=weak,right=none"


def _run(tmp_path_factory, ranks=1, script="mixed_bc_mms_mpi.py", **case):
    cwd = tmp_path_factory.mktemp("_".join("%s%s" % (k, str(v).replace("=", "").replace(",", "")) for k, v in sorted(case.items())) + "_r%d" % ranks)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update({k: str(v) for k, v in case.items()})
    script = os.path.join(HERE, script)
    argv = [sys.executable, script] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, script]
    out = launch_output.run(argv, cwd=cwd, env=env, timeout=1800)
    results = [json.loads(line[len("RESULT "):]) for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(results) == 1, out.stdout[-2000:] + out.stderr[-2000:]
    return results[0]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cache = {}

    def go(ranks=1, **case):
        key = (ranks, tuple(sorted(case.items())))
        if key not in cache:
            cache[key] = _run(tmp_path_factory, ranks, **case)
        return cache[key]
    return go


def _rate(a, b, ha, hb):
    return math.log(a / b) / math.log(ha / hb)


@pytest.mark.parametrize("p", [2, 3])
def test_mixed_kinds_converge_at_the_optimal_rate(run, p):
    coarse, fine = run(P=p, NN=9, KINDS=MIXED), run(P=p, NN=17, KINDS=MIXED)
    assert _rate(coarse["L2"], fine["L2"], coarse["h"], fine["h"]) > p + 1 - 0.15, (coarse, fine)
    assert _rate(coarse["H1"], fine["H1"], coarse["h"], fine["h"]) > p - 0.15, (coarse, fine)


def test_a_strong_face_gets_no_weak_term_and_holds_the_data(run):
    r = run(P=2, NN=9, KINDS=MIXED)
    assert r["strongPts"] > 0 and r["weakPts"] > 0 and r["openPts"] > 0
    assert r["bothWeakAndStrong"] == 0                       # the strong wall is not also a weak wall
    assert r["forcedDofsOnRank0"] == 17 and r["eliminatedOnRank0"] == 0   # the 17 dofs of the left wall: rows replaced, still unknowns
    assert r["forcedValueErr"] < 1.0e-13
    assert r["forceStrongConditions"] is True


def test_the_mixed_problem_is_as_accurate_as_the_all_weak_one(run):
    mixed = run(P=3, NN=17, KINDS=MIXED)
    weak = run(P=3, NN=17, KINDS="left=weak,bottom=weak,top=weak,right=none")
    assert mixed["L2"] < 2.0 * weak["L2"] and mixed["H1"] < 2.0 * weak["H1"], (mixed, weak)


@pytest.mark.parametrize("flux", ["iipg", "sipg"])
def test_a_function_that_names_no_kind_stays_weak_exactly(run, flux):
    bare = run(P=2, NN=9, FLUX=flux, KINDS="left=bare,bottom=bare,top=bare,right=none")
    undeclared = run(P=2, NN=9, FLUX=flux, KINDS="left=weak,bottom=weak,top=weak,right=none", DECLARE=0)
    assert bare["L2"] == undeclared["L2"] and bare["H1"] == undeclared["H1"], (bare, undeclared)
    assert bare["strongPts"] == 0 and bare["forcedDofsOnRank0"] == 0 and bare["weakPts"] == undeclared["weakPts"] > 0


@pytest.mark.parametrize("part", ["node", "element"])
@pytest.mark.parametrize("ranks", [2, 3])
def test_ranks_give_the_serial_errors(run, part, ranks):
    s = run(P=3, NN=9, KINDS=MIXED, PART_TYPE=part)
    r = run(ranks=ranks, P=3, NN=9, KINDS=MIXED, PART_TYPE=part)
    assert r["ranks"] == ranks and r["forceStrongConditions"] is True
    assert abs(r["L2"] - s["L2"]) < 1.0e-8 * s["L2"] and abs(r["H1"] - s["H1"]) < 1.0e-8 * s["H1"], (s, r)
    assert r["forcedValueErr"] < 1.0e-13 and r["bothWeakAndStrong"] == 0


def test_poisson_with_a_strong_and_a_weak_wall_matches_the_all_weak_rates(run):
    kinds = "left=strong,right=weak,bottom=weak,top=strong"
    coarse, fine = run(P=3, NN=9, ADV=0, KINDS=kinds), run(P=3, NN=17, ADV=0, KINDS=kinds)
    assert _rate(coarse["L2"], fine["L2"], coarse["h"], fine["h"]) > 4 - 0.15, (coarse, fine)
    assert _rate(coarse["H1"], fine["H1"], coarse["h"], fine["h"]) > 3 - 0.15, (coarse, fine)
    two = run(ranks=2, P=3, NN=9, ADV=0, KINDS=kinds)
    assert abs(two["L2"] - coarse["L2"]) < 1.0e-8 * coarse["L2"]


@pytest.mark.parametrize("flux", ["strong", "keep"])
def test_a_declared_open_face_is_the_outflow_flag_of_one_face(run, flux):
    strong = "left=strong,bottom=strong,top=strong"
    declared = run(P=2, NN=17, FLUX=flux, FLUXBC="noFlow", KINDS=strong + ",right=none")
    old_way = run(P=2, NN=17, FLUX=flux, FLUXBC="outFlow", KINDS=strong, DECLARE=0)
    wall = run(P=2, NN=17, FLUX=flux, FLUXBC="noFlow", KINDS=strong, DECLARE=0)
    assert declared["L2"] == pytest.approx(old_way["L2"], rel=1.0e-10) and declared["H1"] == pytest.approx(old_way["H1"], rel=1.0e-10)
    assert declared["openPts"] > 0 and old_way["openPts"] == 0
    assert wall["L2"] > 100 * declared["L2"], (declared, wall)       # without the declaration the face is a wall
    assert declared["L2"] < 5.0e-4                                   # and the declared problem converges


def test_a_weak_face_under_a_class_without_a_weak_term_is_refused(tmp_path_factory):
    with pytest.raises(AssertionError) as e:
        _run(tmp_path_factory, P=2, NN=5, FLUX="strong", KINDS="left=weak,bottom=strong,top=strong,right=none")
    assert "has no weak Dirichlet term" in str(e.value)


def test_a_kind_in_a_setter_that_was_not_declared_is_refused(tmp_path_factory):
    # the plain setter returns strong(g) faces; nothing downstream would read the kind
    with pytest.raises(AssertionError) as e:
        _run(tmp_path_factory, P=2, NN=5, KINDS=MIXED, DECLARE=0, UNDECLARED_KINDS=1)
    assert "not declared" in str(e.value)


def _tadr(run, ranks=1, **case):
    return run(ranks=ranks, script="tadr_mixed_bc_mpi.py", **case)


def test_tadr_per_face_kinds_match_the_legacy_all_strong_and_all_weak_runs(run):
    weak_right = _tadr(run, KINDS="left=strong,right=weak,top=none,bottom=none")
    all_weak = _tadr(run, DECLARE=0, KINDS="left=bare,right=bare")
    strong_right = _tadr(run, KINDS="left=weak,right=strong,top=none,bottom=none")
    all_strong = _tadr(run, DECLARE=0, ALL="strong", KINDS="left=bare,right=bare")
    assert weak_right["rms"] == pytest.approx(all_weak["rms"], rel=1.0e-6)
    assert strong_right["rms"] == pytest.approx(all_strong["rms"], rel=1.0e-6)
    assert strong_right["rms"] > 1.2 * weak_right["rms"]          # the right wall's kind matters, and it is read per face
    assert weak_right["uMin"] > 0.05 and strong_right["uMin"] == 0.0
    assert weak_right["strongPts"] == weak_right["weakPts"] == 24 and weak_right["forcedDofsOnRank0"] == 9
    assert weak_right["openPts"] > 0 and weak_right["forceStrongConditions"] is True


def test_tadr_two_ranks_give_the_serial_solution(run):
    case = dict(KINDS="left=strong,right=weak,top=none,bottom=none")
    s, r = _tadr(run, **case), _tadr(run, ranks=2, **case)
    assert r["ranks"] == 2
    assert r["rms"] == pytest.approx(s["rms"], rel=1.0e-9) and r["linf"] == pytest.approx(s["linf"], rel=1.0e-9)
