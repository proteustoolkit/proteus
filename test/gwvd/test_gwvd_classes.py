"""The three gwvd step classes beyond what the 2dbox deck reaches, as unit tests on bare instances.

``test_gwvd_2dbox.py`` runs the deck that uses ``TimeIntegration.NonConservativeBackwardEuler``,
``StepControl.Fixed_dt_controller_systemit`` and ``SplitOperator.ISO_fixed_MinModelStep``; it leaves a few lines of each
unrun, the ones that depend on options or state the deck does not set. The instances here are made with ``object.__new__``
(the constructors need a whole model) and given the attributes the method under test reads:

- ``ISO_fixed_MinModelStep.setFromOptions`` reads ``atol_iso`` and ``velocity_Postprocess`` from the options object when it has them and
  resets ``velocity_Postprocess`` otherwise; ``SysNorm`` is the signed maximum of the residuals (the 2012 code's choice, kept for fidelity),
  so a large negative residual does not block convergence;
- ``Fixed_dt_controller_systemit.choose_dt_model`` sets the model step to ``DT`` (and keeps the first one as ``dt_model_last``),
  ``stepExact_model`` leaves the step alone;
- ``NonConservativeBackwardEuler.calculateGeneralizedInterpolationCoefficients`` does nothing without ``integrateInterpolationPoints``, and
  with it gives m_t = dm (u - u_last) / dt and dm_t = dm / dt at the interpolation points.
"""
import numpy
import pytest

from proteus.SplitOperator import ISO_fixed_MinModelStep
from proteus.StepControl import Fixed_dt_controller_systemit
from proteus.TimeIntegration import NonConservativeBackwardEuler


class _Options(object):
    pass


def _iso():
    iso = object.__new__(ISO_fixed_MinModelStep)
    iso.atol_iso, iso.velocity_Postprocess = 1.0e-8, True
    return iso


def test_iso_reads_its_options():
    iso = _iso()
    options = _Options()
    iso.setFromOptions(options)
    assert iso.atol_iso == 1.0e-8 and iso.velocity_Postprocess is False
    options.atol_iso, options.velocity_Postprocess = 3.0e-5, True
    iso.setFromOptions(options)
    assert iso.atol_iso == 3.0e-5 and iso.velocity_Postprocess is True


def test_iso_norm_is_the_signed_maximum(opts):
    # opts starts Proteus (the MPI communicator globalMax reduces over)
    iso = _iso()
    assert iso.SysNorm([numpy.array([-5.0, -3.0]), numpy.array([1.0e-9, -2.0])]) == 1.0e-9
    assert iso.SysNorm([numpy.array([2.0, -3.0])]) == 2.0


def test_fixed_dt_controller_sets_the_model_step():
    log = []
    c = object.__new__(Fixed_dt_controller_systemit)
    c.dt, c.dt_model_last, c.t_model = 0.25, None, 3.5
    c.saveSolution = lambda: log.append("save")
    c.set_dt_allLevels = lambda: log.append("set_dt")
    c.setSubsteps = lambda substeps: log.append(("substeps", list(substeps)))
    c.choose_dt_model()
    assert c.dt_model == 0.25 and c.dt_model_last == 0.25 and c.solverFailures == 0 and c.errorFailures == 0
    assert log == ["save", "set_dt", ("substeps", [3.5])]
    c.dt = 0.5
    c.choose_dt_model()
    assert c.dt_model == 0.5 and c.dt_model_last == 0.25, "the first step stays dt_model_last"
    c.stepExact_model(10.0)
    assert c.dt_model == 0.5


def _backward_euler(integrate):
    t = object.__new__(NonConservativeBackwardEuler)
    t.integrateInterpolationPoints, t.dt, t.nc = integrate, 0.5, 2
    t.m_ip_last = {0: numpy.array([[1.0, 2.0]]), 1: numpy.array([[0.5, 0.5]])}
    t.m_ip_tmp = {0: numpy.zeros((1, 2)), 1: numpy.zeros((1, 2))}
    return t


def _cip():
    cip = {}
    for ci in range(2):
        cip[('u', ci)] = numpy.array([[3.0 + ci, 4.0 + ci]])
        cip[('mt', ci)] = numpy.zeros((1, 2))
        for cj in range(2):
            cip[('dm', ci, cj)] = numpy.full((1, 2), 2.0 + ci + 10 * cj)
    cip[('dmt', 0, 0)] = numpy.zeros((1, 2))
    cip[('dmt', 1, 0)] = numpy.zeros((1, 2))
    cip[('dm_sge', 0, 1)] = numpy.full((1, 2), 6.0)
    cip[('dmt_sge', 0, 1)] = numpy.zeros((1, 2))
    return cip


def test_interpolation_point_coefficients_are_skipped_unless_asked_for():
    t, cip = _backward_euler(False), _cip()
    before = {k: v.copy() for k, v in cip.items()}
    t.calculateGeneralizedInterpolationCoefficients(cip)
    assert all(numpy.array_equal(cip[k], before[k]) for k in cip)
    assert all(not v.any() for v in t.m_ip_tmp.values())


def test_interpolation_point_coefficients_are_the_non_conservative_time_derivative():
    t, cip = _backward_euler(True), _cip()
    t.calculateGeneralizedInterpolationCoefficients(cip)
    # m_t = dm (u - u_last) / dt with the diagonal dm
    assert numpy.allclose(cip[('mt', 0)], 2.0 * (numpy.array([[3.0, 4.0]]) - numpy.array([[1.0, 2.0]])) / 0.5)
    assert numpy.allclose(cip[('mt', 1)], 13.0 * (numpy.array([[4.0, 5.0]]) - numpy.array([[0.5, 0.5]])) / 0.5)
    assert numpy.allclose(cip[('dmt', 0, 0)], 2.0 / 0.5) and numpy.allclose(cip[('dmt', 1, 0)], 3.0 / 0.5)
    assert numpy.allclose(cip[('dmt_sge', 0, 1)], 6.0 / 0.5)
    assert numpy.array_equal(t.m_ip_tmp[0], cip[('u', 0)]) and numpy.array_equal(t.m_ip_tmp[1], cip[('u', 1)])
