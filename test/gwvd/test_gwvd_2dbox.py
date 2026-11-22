"""The 2-D box deck of the variable-density groundwater flow and transport port (gwvd), run serially for 10 days.

The deck (``2dbox/gwvd_2dbox_so_cg.py``) pairs the flow and transport models of Povich et al. (2013) with the
iterated split-operator step (``SplitOperator.ISO_fixed_MinModelStep``), the fixed-dt controller
(``StepControl.Fixed_dt_controller_systemit``) and the non-conservative backward Euler
(``TimeIntegration.NonConservativeBackwardEuler``).  Those classes and the two coefficient classes
(``TransportCoefficients.VariableDensity_mod2_FlowEquationCoefficients``, ``VDGroundwater_mod2_TransportCoefficients``)
lived in ``proteus/gwvd`` until C49 R1 moved them into the modules named here; the numbers pinned below are the ones
the run gave before the move (bit for bit: the L2 norms of the final head and mass fraction at 17 digits were equal).

Two cases: the plain velocity (``GWVD_CV=0``) and the Knabner-Frolkovic consistent velocity (``GWVD_CV=1``), which
removes most of the mass-fraction overshoot above 1 (1.028 to 1.004).
"""
import os
import shutil
import subprocess
import sys

import h5py
import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "2dbox")

# case -> (norm of h at 10 d, norm of mf at 10 d, max of mf at 10 d)
PINNED = {
    0: (1810.4542075205875, 31.96240250334426, 1.0283524657346117),
    1: (1814.1829661317959, 31.958028300869735, 1.0037795569844588),
}


def _parun():
    found = shutil.which("parun", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    return [found] if found else [sys.executable, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts",
                                                               "parun")]


def _run_deck(cv, tmp_path):
    work = tmp_path / ("cv%d" % cv)
    shutil.copytree(DECK, str(work))
    env = dict(os.environ, GWVD_T_DAYS="10", GWVD_CV=str(cv), OMP_NUM_THREADS="1")
    ran = subprocess.run(_parun() + ["gwvd_2dbox_so_cg.py", "-l1"], cwd=str(work), capture_output=True, text=True,
                         timeout=600, env=env)
    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    (archive,) = [f for f in os.listdir(str(work)) if f.endswith(".h5")]
    with h5py.File(str(work / archive), "r") as f:
        return f["h_t1"][:], f["mf_t1"][:]


@pytest.mark.parametrize("cv", [0, 1])
def test_the_2dbox_deck_runs_and_gives_the_pinned_solution(cv, tmp_path):
    h, mf = _run_deck(cv, tmp_path)
    h_norm, mf_norm, mf_max = PINNED[cv]
    assert numpy.isfinite(h).all() and numpy.isfinite(mf).all()
    assert numpy.linalg.norm(h) == pytest.approx(h_norm, rel=1e-8)
    assert numpy.linalg.norm(mf) == pytest.approx(mf_norm, rel=1e-8)
    assert mf.max() == pytest.approx(mf_max, rel=1e-8)


def test_the_gwvd_classes_live_where_the_decks_import_them():
    from proteus import StepControl, SplitOperator, TimeIntegration, TransportCoefficients
    assert issubclass(StepControl.Fixed_dt_controller_systemit, StepControl.Min_dt_controller)
    assert issubclass(TimeIntegration.NonConservativeBackwardEuler, TimeIntegration.BackwardEuler)
    assert issubclass(SplitOperator.ISO_fixed_MinModelStep, SplitOperator.ISO_fixed_MinAdaptiveModelStep)
    assert issubclass(TransportCoefficients.VariableDensity_mod2_FlowEquationCoefficients,
                      TransportCoefficients.TC_base)
    assert issubclass(TransportCoefficients.VDGroundwater_mod2_TransportCoefficients, TransportCoefficients.TC_base)
    with pytest.raises(ImportError):
        import proteus.gwvd  # noqa: F401  (the directory is gone; gwvd was never installed, so no alias)
