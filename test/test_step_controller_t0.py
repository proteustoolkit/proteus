"""proteustoolkit/proteus#935: initialize_dt_model(t0, tOut) must start the model's clock at t0.

Min_dt_controller and the controllers built on it computed dt from the time integrator or the CFL
number and left t_model_last at its constructor value, 0.0, so a model initialized at t0 > 0 (a cold
start with tnList[0] > 0; a hot start of a model whose controller was not told the restart time) got
t_model = dt instead of t0 + dt."""
import types

import numpy as np
import pytest

from proteus import Comm, StepControl

Comm.get()  # globalMax in the CFL controller needs an initialized communicator


class _TI(object):
    nStages = 1

    def __init__(self):
        self.dt = None
        self.t = None
        self.substeps = None

    def setFromOptions(self, nOptions):
        pass

    def initialize_dt(self, t0, tOut, q):
        self.dt = 0.01

    def set_dt(self, dt):
        self.dt = dt

    def generateSubsteps(self, tList):
        self.substeps = list(tList)


def _model():
    lm = types.SimpleNamespace(timeIntegration=_TI(), nc=1, q={('cfl', 0): np.full(4, 2.0)}, name="stub")
    return types.SimpleNamespace(levelModelList=[lm], uList=[], rList=[], name="stub")


def _options():
    return types.SimpleNamespace(maxSolverFailures=1, maxErrorFailures=1, runCFL=0.5)


@pytest.mark.parametrize("name", ["Min_dt_controller", "Min_dt_RKcontroller", "Min_dt_cfl_controller",
                                  "Min_dt_controller_FCT", "SC_base"])
def test_initialize_dt_model_starts_the_clock_at_t0(name):
    ctl = getattr(StepControl, name)(_model(), _options())
    ctl.initialize_dt_model(0.5, 0.505)
    assert ctl.t_model_last == 0.5
    assert ctl.t_model == pytest.approx(0.5 + ctl.dt_model)
