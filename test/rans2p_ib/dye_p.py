"""A dye in the beam channel (VOF with no level set, carried by RANS2P_IB's velocity): dye 1 enters at x = 0 over
the lower half of the inflow, the channel starts clean."""
from proteus.default_p import *
from proteus.mprans import VOF
from beam_channel_p import nd, L, T, eps, _inflow, _outflow

name = "dye"
LevelModelType = VOF.LevelModel
coefficients = VOF.Coefficients(LS_model=None, V_model=0, RD_model=None, ME_model=1, checkMass=False, useMetrics=1.0,
                                epsFact=0.0, sc_uref=1.0, sc_beta=1.0)


def getDBC(x, flag):
    if _inflow(x):
        return lambda x, t: 1.0 if x[2] < 0.5 * L[2] else 0.0


dirichletConditions = {0: getDBC}


def getAFBC(x, flag):
    if not (_inflow(x) or _outflow(x)):
        return lambda x, t: 0.0


advectiveFluxBoundaryConditions = {0: getAFBC}
diffusiveFluxBoundaryConditions = {0: {}}


class Clean(object):
    def uOfXT(self, x, t):
        return 0.0


initialConditions = {0: Clean()}
