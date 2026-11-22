"""A hot-start state written before the kernels moved (C49 R1) must still continue.

The step policy a state carries holds class paths (``type(obj).__module__``). The kernel packages (proteus.mprans,
proteus.richards, proteus.m_comp_co2, proteus.elastoplastic) became proteus.kernels, so the same class reports a new
path; ``NS_base._normalize_archived_policy`` reads the old ones as the current ones.
"""
import pytest

from proteus.NumericalSolution import NS_base, _current_module_path


@pytest.mark.parametrize("old, new", [
    ("proteus.mprans.NCLS.RKEV lag=True", "proteus.kernels.NCLS.RKEV lag=True"),
    ("proteus.mprans.VOF.RKEV lag=None", "proteus.kernels.VOF.RKEV lag=None"),
    ("proteus.richards.Richards.Coefficients order=2", "proteus.kernels.Richards.Coefficients order=2"),
    ("proteus.m_comp_co2.m_comp_co2.X", "proteus.kernels.m_comp_co2.X"),
    ("proteus.elastoplastic.ElastoPlastic.Y", "proteus.kernels.ElastoPlastic.Y"),
    ("proteus.ADR.Coefficients", "proteus.kernels.ADR.Coefficients"),
    # modules that left the kernel package go to their own homes, not to proteus.kernels
    ("proteus.mprans.BodyDynamics.RigidBody", "proteus.mbd.BodyDynamics.RigidBody"),
    ("proteus.mprans.MoveMeshMonitor.Coefficients", "proteus.MoveMeshMonitor.Coefficients"),
    # a current path, another package, and a name that merely starts like a package are left alone
    ("proteus.kernels.NCLS.RKEV lag=True", "proteus.kernels.NCLS.RKEV lag=True"),
    ("proteus.TimeIntegration.BackwardEuler order=1", "proteus.TimeIntegration.BackwardEuler order=1"),
    ("proteus.mprans_extra.X", "proteus.mprans_extra.X"),
    ("proteus.ADRProblem.X", "proteus.ADRProblem.X"),
])
def test_old_module_paths_read_as_the_current_ones(old, new):
    assert _current_module_path(old) == new


def test_an_archived_policy_is_normalized_value_by_value():
    old = {"timeIntegration": "proteus.TimeIntegration.BackwardEuler order=1",
           "stepController": "Min_dt_controller",
           "shockCapturing": "proteus.mprans.NCLS.RKEV lag=True",
           "stabilization": None,
           "DT": 0.01,
           "atol_u": {"0": 1e-8}}
    now = NS_base._normalize_archived_policy(old)
    assert now["shockCapturing"] == "proteus.kernels.NCLS.RKEV lag=True"
    assert {k: v for k, v in now.items() if k != "shockCapturing"} == {k: v for k, v in old.items() if k != "shockCapturing"}
