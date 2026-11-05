"""One rigid beam (a vegetation stem) in a 3D water channel: the RANS2P_IB deck.

RANS2P_IB is RANS2P plus the drag of immersed Euler-Bernoulli beams (beamFEM), spread onto the mesh with a smoothed
delta function. Single phase, gravity along -z. Starts from rest: inflow U at x = 0, p = 0 at x = Lx, no slip on the
bottom and the side walls, free slip on top. The beam stands at (0.2, 0.345) from z = 0, inside the drag-averaging box
y in [0.1725, 0.5175] that RANS2P_IB hard-codes. Context option beamRigid=False makes it a soft flexible beam (it
bends about 1 cm in 0.04 s).
"""
from proteus import *
from proteus.default_p import *
from proteus import Context
from proteus.mprans import RANS2P_IB

opts = Context.Options([("beamRigid", True, "a rigid beam, or a flexible one (EI = GJ = 1e-4)")])

nd = 3
L = (0.4, 0.69, 0.2)
domain = None
name = "beam_channel"
T = 0.04
U = 0.1
rho_0, nu_0 = 998.2, 1.0e-3
g = [0.0, 0.0, -9.8]
eps = 1.0e-8

LevelModelType = RANS2P_IB.LevelModel
coefficients = RANS2P_IB.Coefficients(epsFact=1.5, rho_0=rho_0, nu_0=nu_0, rho_1=rho_0, nu_1=nu_0, g=g, nd=nd,
                                      LS_model=None, forceStrongDirichlet=False, useMetrics=1.0,
                                      beamLocation=[(0.2, 0.345)], beamLength=[0.1], beamRadius=[0.005],
                                      EI=[1.0e3 if opts.beamRigid else 1.0e-4],
                                      GJ=[1.0e3 if opts.beamRigid else 1.0e-4], nBeamElements=4, beam_quadOrder=3,
                                      beamRigid=opts.beamRigid, beamFilename="Beams")


def _inflow(x):
    return x[0] < eps


def _outflow(x):
    return x[0] > L[0] - eps


def _wall(x):
    return x[1] < eps or x[1] > L[1] - eps or x[2] < eps


def getDBC_p(x, flag):
    if _outflow(x):
        return lambda x, t: 0.0


def getDBC_u(x, flag):
    if _inflow(x):
        return lambda x, t: U
    if _wall(x) and not _outflow(x):
        return lambda x, t: 0.0


def getDBC_vw(x, flag):
    if _inflow(x) or (_wall(x) and not _outflow(x)):
        return lambda x, t: 0.0


dirichletConditions = {0: getDBC_p, 1: getDBC_u, 2: getDBC_vw, 3: getDBC_vw}


def getAFBC_p(x, flag):
    if _inflow(x):
        return lambda x, t: -U
    if not _outflow(x):
        return lambda x, t: 0.0


def getAFBC_vel(x, flag):
    if not (_inflow(x) or _outflow(x)):
        return lambda x, t: 0.0


def getDFBC_vel(x, flag):
    if not (_inflow(x) or _wall(x)):
        return lambda x, t: 0.0


advectiveFluxBoundaryConditions = {0: getAFBC_p, 1: getAFBC_vel, 2: getAFBC_vel, 3: getAFBC_vel}
diffusiveFluxBoundaryConditions = {0: {}, 1: {1: getDFBC_vel}, 2: {2: getDFBC_vel}, 3: {3: getDFBC_vel}}


class Zero(object):
    def uOfXT(self, x, t):
        return 0.0


initialConditions = {0: Zero(), 1: Zero(), 2: Zero(), 3: Zero()}
