"""Per-face strong | weak | none boundary kinds (C12) in RANS2P, 2D and 3D, on a manufactured Navier-Stokes solution.

`rans2p_mixed_bc_mms_mpi.py` solves single phase Navier-Stokes (P1/P1 with the model's stabilization, nu = 0.1) on the unit
square or cube against a divergence-free manufactured field whose velocity on the wall x = 1 carries no stress, with a
DIFFERENT kind on each wall: the velocity is strong on the left and top (and back, in 3D), weak on the bottom (and front), and
declared open ('none') on the right, where the pressure is the weak datum. The exact solution is the data of every strong and
weak wall.

* the mixed problem converges at the rates of the element (u: L2 2, H1 1; p: L2 about 1.6) and is as accurate as the all-weak one;
* a strong face gets no weak term from the flux class (no face point is both) and the dofs of the strong walls hold the data to
  round-off (their rows are replaced, they stay unknowns);
* every face declared strong reproduces the legacy ``forceStrongDirichlet`` model bit for bit; a function that names no kind in
  a declared setter stays weak, bit for bit the undeclared run;
* serial = 2 and 3 ranks (node and element partitions);
* ``forceStrongDirichlet`` together with a declared setter is refused.

RANS2P's kernels have no kind flag of their own: they read ``isDOFBoundary`` per component, which ``kernelIsDOFBoundary`` fills with the
weak faces and the strong ones (the datum a strong face keeps imposing, as ``forceStrongDirichlet`` always did).
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
MIXED2 = "left=strong,bottom=weak,top=strong,right=none"
MIXED3 = MIXED2 + ",front=weak,back=strong"
ALLWEAK2 = "left=weak,bottom=weak,top=weak,right=none"
ALLSTRONG2 = "left=strong,bottom=strong,top=strong,right=none"
ALLSTRONG3 = ALLSTRONG2 + ",front=strong,back=strong"
BARE2 = "left=bare,bottom=bare,top=bare,right=none"


def _run(tmp_path_factory, ranks=1, **case):
    cwd = tmp_path_factory.mktemp("_".join("%s%s" % (k, str(v).replace("=", "").replace(",", "")) for k, v in sorted(case.items())) + "_r%d" % ranks)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["PETSC_OPTIONS"] = "-mat_partitioning_type current"
    env.update({k: str(v) for k, v in case.items()})
    script = os.path.join(HERE, "rans2p_mixed_bc_mms_mpi.py")
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


def test_mixed_kinds_converge_at_the_rates_of_the_element_2d(run):
    mid, fine = run(ND=2, NN=17, KINDS=MIXED2), run(ND=2, NN=33, KINDS=MIXED2)
    for k in ("u0", "u1"):
        assert _rate(mid["L2"][k], fine["L2"][k], mid["h"], fine["h"]) > 1.7, (mid, fine)
        assert _rate(mid["H1"][k], fine["H1"][k], mid["h"], fine["h"]) > 0.9, (mid, fine)
    assert _rate(mid["L2"]["p"], fine["L2"]["p"], mid["h"], fine["h"]) > 1.4, (mid, fine)


def test_the_mixed_problem_is_as_accurate_as_the_all_weak_one_2d(run):
    mixed = run(ND=2, NN=33, KINDS=MIXED2)
    weak = run(ND=2, NN=33, KINDS=ALLWEAK2)
    for k in mixed["L2"]:
        assert mixed["L2"][k] < 2.0 * weak["L2"][k] and mixed["H1"][k] < 1.5 * weak["H1"][k], (mixed, weak)


def test_a_strong_face_gets_no_weak_term_and_holds_the_data_2d(run):
    r = run(ND=2, NN=17, KINDS=MIXED2)
    for ci in ("1", "2"):      # u and v
        f = r["faces"][ci]
        assert f["strong"] > 0 and f["weak"] > 0 and f["open"] > 0 and f["both"] == 0
    assert r["faces"]["0"]["weak"] > 0 and r["faces"]["0"]["strong"] == 0     # the pressure datum of the right wall
    assert r["forcedDofs"] == 66 and r["forcedValueErr"] == 0.0     # 33 dofs of the left wall + 33 of the top, for u and v, corners shared
    assert r["forceStrongConditions"] is True


def test_every_face_strong_is_the_legacy_force_strong_model_2d(run):
    declared = run(ND=2, NN=17, KINDS=ALLSTRONG2, PKINDS="right=strong")
    legacy = run(ND=2, NN=17, KINDS=ALLSTRONG2, PKINDS="right=strong", DECLARE=0, ALL="strong")
    assert declared["L2"] == legacy["L2"] and declared["H1"] == legacy["H1"], (declared, legacy)
    assert declared["forcedDofs"] == legacy["forcedDofs"] > 0


def test_a_function_that_names_no_kind_stays_weak_exactly_2d(run):
    bare = run(ND=2, NN=17, KINDS=BARE2, PKINDS="right=bare")
    undeclared = run(ND=2, NN=17, KINDS=BARE2, PKINDS="right=bare", DECLARE=0)
    assert bare["L2"] == undeclared["L2"] and bare["H1"] == undeclared["H1"], (bare, undeclared)
    assert bare["forcedDofs"] == 0 and bare["faces"]["1"]["strong"] == 0


@pytest.mark.parametrize("part", ["node", "element"])
@pytest.mark.parametrize("ranks", [2, 3])
def test_ranks_give_the_serial_errors_2d(run, part, ranks):
    s = run(ND=2, NN=17, KINDS=MIXED2, PART_TYPE=part)
    r = run(ranks=ranks, ND=2, NN=17, KINDS=MIXED2, PART_TYPE=part)
    assert r["ranks"] == ranks
    for q in ("L2", "H1"):
        for k in s[q]:
            assert abs(r[q][k] - s[q][k]) < 1.0e-9 * s[q][k], (q, k, s, r)
    assert r["forcedValueErr"] == 0.0


def test_mixed_kinds_converge_and_match_the_serial_run_in_3d(run):
    coarse, fine = run(ND=3, NN=9, KINDS=MIXED3), run(ND=3, NN=17, KINDS=MIXED3)
    for k in ("u0", "u1"):
        assert _rate(coarse["L2"][k], fine["L2"][k], coarse["h"], fine["h"]) > 1.5, (coarse, fine)
        assert _rate(coarse["H1"][k], fine["H1"][k], coarse["h"], fine["h"]) > 0.9, (coarse, fine)
    assert _rate(coarse["L2"]["p"], fine["L2"]["p"], coarse["h"], fine["h"]) > 1.0, (coarse, fine)
    weak = run(ND=3, NN=9, KINDS=ALLWEAK2 + ",front=weak,back=weak")
    for k in coarse["L2"]:
        assert coarse["L2"][k] < 2.0 * weak["L2"][k], (coarse, weak)
    for ci in ("1", "2", "3"):
        f = fine["faces"][ci]
        assert f["strong"] > 0 and f["weak"] > 0 and f["open"] > 0 and f["both"] == 0
    two = run(ranks=2, ND=3, NN=9, KINDS=MIXED3)
    for q in ("L2", "H1"):
        for k in coarse[q]:
            assert abs(two[q][k] - coarse[q][k]) < 1.0e-9 * coarse[q][k], (q, k, coarse, two)
    assert two["forcedValueErr"] == 0.0


def test_every_face_strong_is_the_legacy_force_strong_model_3d(run):
    declared = run(ND=3, NN=5, KINDS=ALLSTRONG3, PKINDS="right=strong")
    legacy = run(ND=3, NN=5, KINDS=ALLSTRONG3, PKINDS="right=strong", DECLARE=0, ALL="strong")
    assert declared["L2"] == legacy["L2"] and declared["H1"] == legacy["H1"], (declared, legacy)


def test_force_strong_dirichlet_with_a_declared_setter_is_refused(tmp_path_factory):
    with pytest.raises(AssertionError) as e:
        _run(tmp_path_factory, ND=2, NN=5, KINDS=MIXED2, ALL="strong")
    assert "forceStrongDirichlet makes every Dirichlet face strong" in str(e.value)


def test_a_declared_velocity_setter_with_a_bare_undeclared_pressure_setter_is_the_weak_pressure_2d(run):
    undeclared_p = run(ND=2, NN=17, KINDS=MIXED2, PKINDS="right=bare", PDECLARE=0)
    declared_weak_p = run(ND=2, NN=17, KINDS=MIXED2, PKINDS="right=weak")
    assert undeclared_p["L2"] == declared_weak_p["L2"] and undeclared_p["H1"] == declared_weak_p["H1"], (undeclared_p, declared_weak_p)
    assert undeclared_p["forcedDofs"] == declared_weak_p["forcedDofs"] > 0 and undeclared_p["faces"]["0"]["strong"] == 0


def test_the_constant_pressure_null_space_flag_follows_a_strong_pressure_datum(run):
    # read at the first Jacobian (zero velocity). Old behaviour, kept: forceStrongDirichlet never declares the null space; any
    # other problem (a weak or absent pressure datum) declares it while the velocity is zero. A declared setter with a strongly
    # imposed pressure datum is what forceStrongDirichlet's False stood for.
    weak_vel = "left=weak,bottom=weak,top=weak,right=none"
    assert run(ND=2, NN=9, KINDS=weak_vel, PKINDS="right=weak")["nullSpaceAtFirstJacobian"] is True
    assert run(ND=2, NN=9, KINDS=weak_vel, PKINDS="right=weak", DECLARE=0)["nullSpaceAtFirstJacobian"] is True
    assert run(ND=2, NN=9, KINDS=weak_vel, PKINDS="right=strong")["nullSpaceAtFirstJacobian"] is False
    legacy = run(ND=2, NN=9, KINDS=ALLSTRONG2, PKINDS="right=strong", DECLARE=0, ALL="strong")
    assert legacy["nullSpaceAtFirstJacobian"] is False
    assert run(ranks=2, ND=2, NN=9, KINDS=weak_vel, PKINDS="right=strong")["nullSpaceAtFirstJacobian"] is False  # collective: both ranks
