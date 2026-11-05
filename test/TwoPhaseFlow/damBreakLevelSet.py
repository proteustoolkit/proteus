"""damBreak.py with the level-set family instead of CLSVOF + projection: RANS2P, VOF, NCLS, RDLS and MCorr
(useDefaultModels(flowModel=0, interfaceModel=0)), the boundary conditions from domain.bc. Same geometry, mesh
and options; used by test_step_retry.py. interface=1 runs RANS2P with CLSVOF instead (interfaceModel=1);
turbulence=1 adds the k-epsilon models (Kappa, Dissipation).
"""
"""
danbreak 2-D
"""
import numpy as np
from proteus import (Domain, Context)
from proteus.Profiling import logEvent
from proteus.mprans.SpatialTools import Tank2D
from proteus.mprans import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
import proteus.TwoPhaseFlow.utils.Parameters as Parameters
from proteus.Gauges import PointGauges, LineIntegralGauges, LineGauges
import os

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts= Context.Options([
    ("final_time",3.0,"Final time for simulation"),
    ("dt_output",0.01,"Time interval to output solution"),
    ("cfl",0.25,"Desired CFL restriction"),
    ("he",0.01,"he relative to Length of domain in x"),
    ("refinement",3,"level of refinement"),
    ("hotstart",False,"will be hotstartting?"),
    ("genMesh", False,"generate the mesh on the fly"),
    ("interface", 0, "0: VOF, NCLS, RDLS and MCorr; 1: CLSVOF"),
    ("turbulence", 0, "1: add the k-epsilon models (Kappa, Dissipation; useRANS=1)")
    ])

# ****************** #
# ***** GAUGES ***** #
# ****************** #
height_gauges1 = LineGauges(gauges=((("phi",),
                                        (((2.724, 0.0, 0.0),
                                          (2.724, 1.8, 0.0)), # We consider this one in our paper
                                         ((2.228, 0.0, 0.0),
                                          (2.228, 1.8, 0.0)), # We consider this one in our paper
                                         ((1.732, 0.0, 0.0),
                                          (1.732, 1.8, 0.0)),
                                         ((0.582, 0.0, 0.0),
	                                  (0.582, 1.8, 0.0)))),),
                                        fileName="height1.csv")

height_gauges2 = LineGauges(gauges=((("phi",),
                                     (((0.0, 0.0, 0.0),
                                       (0.0, 0.0, -0.01)),
                                      ((0.0, 0.0, 0.0),
                                        (3.22, 0.0, 0.0)))),),
                            fileName="height2.csv")

pressure_gauges = PointGauges(gauges=((('p',),
                                      ((3.22, 0.16, 0.0), #P1
                                       (3.22, 0.584, 0.0), #P3
                                       (3.22, 0.12, 0.0))),), # This is the one considered in our paper
                                       fileName="pressure.csv")


# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
tank_dim = (3.22,1.8)
refinement = opts.refinement
structured=False
if structured:
    nny = 5*(2**refinement)+1
    nnx = 2*(nnx-1)+1
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags
    triangleFlag=1
else:
    nnx = nny = None
    domain = Domain.PlanarStraightLineGraphDomain()

# ----- TANK ----- #
tank = Tank2D(domain, tank_dim)

# ----- EXTRA BOUNDARY CONDITIONS ----- #
tank.BC['y+'].setAtmosphere()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()

he = tank_dim[0]*opts.he
domain.MeshOptions.he = he
st.assembleDomain(domain)
domain.MeshOptions.triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)
domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"meshDambreak"
domain.MeshOptions.triangleFlag=0
domain.MeshOptions.genMesh=opts.genMesh
if opts.genMesh:
    domain.writePoly("meshDambreak")

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self,x,t):
        return 0.0

waterLine_y = 0.6
waterLine_x = 1.2
############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.domain=domain

myTpFlowProblem.outputStepping.final_time = opts.final_time
myTpFlowProblem.outputStepping.dt_output = opts.dt_output
myTpFlowProblem.outputStepping.systemStepExact = True

if opts.hotstart:
    myTpFlowProblem.outputStepping.systemStepExact = False
else:
    myTpFlowProblem.outputStepping.systemStepExact = True



myTpFlowProblem.SystemPhysics.setDefaults()
myTpFlowProblem.SystemPhysics.useDefaultModels(flowModel=0,interfaceModel=opts.interface)
if opts.turbulence:
    from proteus.TwoPhaseFlow.utils import Parameters
    myTpFlowProblem.SystemPhysics.useRANS = 1
    myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelKappa, 'kappa')
    myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelDissipation, 'dissipation')
myTpFlowProblem.SystemNumerics.cfl=opts.cfl
myTpFlowProblem.SystemNumerics.useSuperlu=True

import numpy as npls
class phi_init(object):
    def uOfXT(self, x, t):
        dx, dy = x[0] - waterLine_x, x[1] - waterLine_y
        if dx < 0 and dy < 0:
            return max(dx, dy)
        if dx > 0 and dy > 0:
            return npls.hypot(dx, dy)
        return max(dx, dy)
class vof_init(object):
    def uOfXT(self, x, t):
        return smoothedHeaviside(1.5 * he, phi_init().uOfXT(x, t))
from proteus.ctransportCoefficients import smoothedHeaviside
m = myTpFlowProblem.SystemPhysics.modelDict
m['flow'].p.initialConditions['p'] = zero()
m['flow'].p.initialConditions['u'] = zero()
m['flow'].p.initialConditions['v'] = zero()
if opts.interface == 0:
    m['vof'].p.initialConditions['vof'] = vof_init()
    m['ncls'].p.initialConditions['phi'] = phi_init()
    m['rdls'].p.initialConditions['phid'] = phi_init()
    m['mcorr'].p.initialConditions['phiCorr'] = zero()
else:
    m['clsvof'].p.initialConditions['clsvof'] = phi_init()
if opts.turbulence:
    class constant(object):
        def __init__(self, value):
            self.value = value
        def uOfXT(self, x, t):
            return self.value
    m['kappa'].p.initialConditions['kappa'] = constant(1.0e-4)
    m['dissipation'].p.initialConditions['epsilon'] = constant(1.0e-4)

myTpFlowProblem.SystemPhysics.gravity = np.array([0.0,-9.8,0.0])

m = myTpFlowProblem.SystemPhysics.modelDict
m['flow'].n.ShockCapturingOptions.shockCapturingFactor = 0.5
m['flow'].p.coefficients.useVF=1.0
m['flow'].p.coefficients.eb_penalty_constant = 1e6


