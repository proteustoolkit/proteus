"""A 3D dam break with the level-set family: RANS2P, VOF, NCLS, RDLS and MCorr (useDefaultModels(flowModel=0,
interfaceModel=0)), as damBreakLevelSet.py in 2D. The water column is a block in one corner of the tank (x < 1.2,
y < 0.5, z < 0.6 in a 3.22 x 1.0 x 1.8 tank), so the flow is three-dimensional from the first step. Gravity is -z, the
top is open (atmosphere), the other walls are free-slip. The tetrahedral mesh is generated at load (TetGen, he).

The 3D RANS2P case for TwoPhaseFlow's RANS2P preconditioner choice (C19 round 3): the velocity block there has three
interlaced components.
"""
import os

import numpy as np

from proteus import Comm, Context, Domain
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.mprans import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow

opts = Context.Options([
    ("final_time", 3.0, "Final time for simulation"),
    ("dt_output", 0.01, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("he", 0.1, "Maximum element diameter (TetGen's volume constraint is he**3/6)"),
    ("useSuperlu", False, "superlu_dist direct solves for every model instead of the iterative defaults"),
    ])

tank_dim = (3.22, 1.0, 1.8)
column = (1.2, 0.5, 0.6)

domain = Domain.PiecewiseLinearComplexDomain()
tank = st.Tank3D(domain, tank_dim)
tank.BC['z+'].setAtmosphere()
for side in ('x-', 'x+', 'y-', 'y+', 'z-'):
    tank.BC[side].setFreeSlip()

he = opts.he
domain.MeshOptions.he = he
st.assembleDomain(domain)
domain.MeshOptions.triangleOptions = "VApq1.35q12feena%e" % (he**3/6.0,)
domain.MeshOptions.genMesh = True
domain.polyfile = os.path.join(os.getcwd(), "meshDamBreak3D")
if Comm.get().isMaster():  # TetGen runs on rank 0; one writer, not one per rank
    domain.writePoly("meshDamBreak3D")


class zero(object):
    def uOfXT(self, x, t):
        return 0.0


class phi_init(object):
    """Signed distance to the water column (negative in the water)."""
    def uOfXT(self, x, t):
        d = np.array([x[0] - column[0], x[1] - column[1], x[2] - column[2]])
        if (d <= 0.0).all():
            return float(d.max())
        return float(np.sqrt((np.maximum(d, 0.0)**2).sum()))


class vof_init(object):
    def uOfXT(self, x, t):
        return smoothedHeaviside(1.5*he, phi_init().uOfXT(x, t))


myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.domain = domain
myTpFlowProblem.outputStepping.final_time = opts.final_time
myTpFlowProblem.outputStepping.dt_output = opts.dt_output
myTpFlowProblem.outputStepping.systemStepExact = True

myTpFlowProblem.SystemPhysics.setDefaults()
myTpFlowProblem.SystemPhysics.useDefaultModels(flowModel=0, interfaceModel=0)
myTpFlowProblem.SystemPhysics.gravity = np.array([0.0, 0.0, -9.8])
myTpFlowProblem.SystemNumerics.cfl = opts.cfl
myTpFlowProblem.SystemNumerics.useSuperlu = opts.useSuperlu

m = myTpFlowProblem.SystemPhysics.modelDict
for name in ('p', 'u', 'v', 'w'):
    m['flow'].p.initialConditions[name] = zero()
m['vof'].p.initialConditions['vof'] = vof_init()
m['ncls'].p.initialConditions['phi'] = phi_init()
m['rdls'].p.initialConditions['phid'] = phi_init()
m['mcorr'].p.initialConditions['phiCorr'] = zero()
m['flow'].n.ShockCapturingOptions.shockCapturingFactor = 0.5
m['flow'].p.coefficients.useVF = 1.0
m['flow'].p.coefficients.eb_penalty_constant = 1e6
