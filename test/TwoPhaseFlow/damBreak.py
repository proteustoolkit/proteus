"""
danbreak 2-D
"""
import numpy as np
from proteus import (Domain, Context)
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D
from proteus import SpatialTools as st
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
    ("genMesh", False,"generate the mesh on the fly")
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
class clsvof_init_cond(object):
    def uOfXT(self,x,t):
        if x[0] < waterLine_x and x[1] < waterLine_y:
            return -1.0
        elif x[0] > waterLine_x or x[1] > waterLine_y:
            return 1.0
        else:
            return 0.0

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################

boundaryConditions = {
    # DIRICHLET BCs #
    'pressure_DBC': lambda x, flag: domain.bc[flag].p_dirichlet.init_cython(),
    'pressure_increment_DBC': lambda x, flag: domain.bc[flag].pInc_dirichlet.init_cython(),
    'vel_u_DBC': lambda x, flag: domain.bc[flag].u_dirichlet.init_cython(),
    'vel_v_DBC': lambda x, flag: domain.bc[flag].v_dirichlet.init_cython(),
    'vel_w_DBC': lambda x, flag: domain.bc[flag].w_dirichlet.init_cython(),
    'clsvof_DBC': lambda x, flag: domain.bc[flag].vof_dirichlet.init_cython(),
    # ADVECTIVE FLUX BCs #
    'pressure_AFBC': lambda x, flag: domain.bc[flag].p_advective.init_cython(),
    'pressure_increment_AFBC': lambda x, flag: domain.bc[flag].pInc_advective.init_cython(),
    'vel_u_AFBC': lambda x, flag: domain.bc[flag].u_advective.init_cython(),
    'vel_v_AFBC': lambda x, flag: domain.bc[flag].v_advective.init_cython(),
    'vel_w_AFBC': lambda x, flag: domain.bc[flag].w_advective.init_cython(),
    'clsvof_AFBC': lambda x, flag: domain.bc[flag].vof_advective.init_cython(),
    # DIFFUSIVE FLUX BCs #
    'pressure_increment_DFBC': lambda x, flag: domain.bc[flag].pInc_diffusive.init_cython(),
    'vel_u_DFBC': lambda x, flag: domain.bc[flag].u_diffusive.init_cython(),
    'vel_v_DFBC': lambda x, flag: domain.bc[flag].v_diffusive.init_cython(),
    'vel_w_DFBC': lambda x, flag: domain.bc[flag].w_diffusive.init_cython(),
    'clsvof_DFBC': lambda x, flag: None}

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
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelCLSVOF,'clsvof')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelRANS3PF,'flow')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressureIncrement,'pressureInc')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressure,'pressure')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressureInitial,'pressureInit')
myTpFlowProblem.SystemNumerics.cfl=opts.cfl
myTpFlowProblem.SystemNumerics.useSuperlu=True

myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=zero()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=zero()
myTpFlowProblem.SystemPhysics.modelDict['clsvof'].p.initialConditions['clsvof'] = clsvof_init_cond()
myTpFlowProblem.SystemPhysics.modelDict['pressure'].p.initialConditions['p'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInc'].p.initialConditions['pInc'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInit'].p.initialConditions['pInit'] = zero()

myTpFlowProblem.SystemPhysics.boundaryConditions=boundaryConditions
myTpFlowProblem.SystemPhysics.gravity = np.array([0.0,-9.8,0.0])

m = myTpFlowProblem.SystemPhysics.modelDict
m['clsvof'].p.coefficients.disc_ICs = True
m['clsvof'].auxiliaryVariables = [height_gauges1, height_gauges2]
m['pressure'].auxiliaryVariables = [pressure_gauges]
m['flow'].n.ShockCapturingOptions.shockCapturingFactor = 0.5
m['flow'].p.coefficients.useVF=1.0
m['flow'].p.coefficients.eb_penalty_constant = 1e6


