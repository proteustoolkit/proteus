"""The MCorr3P level solvers other than Newton (ElementNewton, ElementConstantNewton, GlobalConstantNewton, DummyNewton) on the 2D rising bubble.

MCorr.py has a deck switch for them (``correctionType`` of the vortex2D deck, test/levelset/vortex2D/test_mcorr_correction_types.py);
MCorr3P.py carries the same four classes and no deck selected any. mcorr3p_correction_mpi.py runs the rising-bubble deck
(RANS3PF, VOF3P, NCLS3P, RDLS, MCorr3P; t = 0.004) with MCorr3P on the solver named by CORRECTION. What is checked:

- 'none', 'dg' and 'dgp0' run: the model used the named class, its dofs are finite;
- 'none' leaves the level set uncorrected (every dof 0);
- Newton (the deck's own) conserves the VOF mass to roundoff (3e-17), as the reference the others are read against.

Two findings, pinned here so that a fix is noticed (report for the kernel's owner in HANDOFF_2026-11-23_c49-r3-done.md):

- 'global' fails in MCorr3P.globalConstantRJ with ``R = np.array(dtype=float)`` (TypeError: array() missing required
  argument 'object'), line 977: the test expects that error and fails when it is fixed;
- 'dg' and 'dgp0' end with the mass error of 'none' (0.0155 against 3e-17 for Newton) and one value in every dof, i.e.
  they apply no correction; the test records that (``test_dg_and_dgp0_apply_no_correction``) and fails when they do.
"""
import importlib.util
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)
HELPER = os.path.join(HERE, "mcorr3p_correction_mpi.py")


def _run(cwd, correction):
    env = dict(os.environ, OMP_NUM_THREADS="1", CORRECTION=correction)
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [HERE, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    launch_output.run([sys.executable, HELPER], cwd=cwd, env=env, timeout=900)
    return json.load(open(os.path.join(str(cwd), "mcorr3p.json")))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for correction in ("newton", "none", "dg", "dgp0"):
        d = tmp_path_factory.mktemp(correction)
        out[correction] = _run(d, correction)
    return out


def test_newton_conserves_the_mass(runs):
    assert runs["newton"]["solver"] == "Newton"
    assert runs["newton"]["mass_error"] < 1e-12


@pytest.mark.parametrize("correction, solver", [("none", "DummyNewton"), ("dg", "ElementNewton"), ("dgp0", "ElementConstantNewton")])
def test_the_named_level_solver_runs(runs, correction, solver):
    assert runs[correction]["solver"] == solver
    assert runs[correction]["finite"]


def test_none_leaves_the_level_set_uncorrected(runs):
    assert runs["none"]["distinct"] == 1


def test_dg_and_dgp0_apply_no_correction(runs):
    # when this fails, ElementNewton / ElementConstantNewton correct the mass: replace it with a conservation test
    for correction in ("dg", "dgp0"):
        assert runs[correction]["distinct"] == 1
        assert runs[correction]["mass_error"] == pytest.approx(runs["none"]["mass_error"], rel=1e-6)


def test_global_correction_fails_in_globalconstantrj(tmp_path):
    # when this fails, MCorr3P.globalConstantRJ works: replace it with a test that the shift conserves the mass
    with pytest.raises(AssertionError, match=r"array\(\) missing required argument 'object'"):
        _run(tmp_path, "global")
