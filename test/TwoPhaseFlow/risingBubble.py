"""
Rising bubble test
"""
import numpy as np
from proteus import (Domain, Context)
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D
from proteus import SpatialTools as st
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
import os

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts= Context.Options([
    ("test_case",1,"Rising bubble test cases: 1,2"),
    ('ns_model',1,"ns_model = {rans2p,rans3p}"),
    ("final_time",3.0,"Final time for simulation"),
    ("dt_output",0.1,"Time interval to output solution"),
    ("cfl",0.2,"Desired CFL restriction"),
    ("refinement",3,"level of refinement"),
    ("ARTIFICIAL_VISCOSITY",3,"artificial viscosity")
    ])

assert opts.ns_model==1, "Surface tension is only implemented with rans3p. use ns_model=1"
assert opts.test_case == 1 or opts.test_case==2, "test_case must be 1 or 2"

# ****************** #
# ***** GAUGES ***** #
# ****************** #
# None

# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
tank_dim = (1.0,2.0)
refinement = opts.refinement
structured=True
if structured:
    nnx = 5*(2**refinement)+1
    #nnx = 4 * refinement**2 +2
    nny = 2*nnx
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags
    he=1.0/(nnx-1)
    domain.MeshOptions.he = he
    domain.MeshOptions.nnx = nnx
    domain.MeshOptions.nny = nny
    domain.MeshOptions.structured = True
    domain.MeshOptions.triangleFlag=1

else:
    vertices = [[0.0, 0.0],  #0
                [tank_dim[0], 0.0],  #1
                [tank_dim[0], tank_dim[1]],  #2
                [0.0, tank_dim[1]]]  #3
    vertexFlags = [boundaryTags['bottom'],
                   boundaryTags['bottom'],
                   boundaryTags['top'],
                   boundaryTags['top']]
    segments = [[0, 1],
                [1, 2],
                [2, 3],
                [3, 0]]
    segmentFlags = [boundaryTags['bottom'],
                    boundaryTags['right'],
                    boundaryTags['top'],
                    boundaryTags['left']]
    regions = [[tank_dim[0]/2., tank_dim[1]/2.]]
    regionFlags = [1]
    domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                                  vertexFlags=vertexFlags,
                                                  segments=segments,
                                                  segmentFlags=segmentFlags,
                                                  regions=regions,
                                                  regionFlags=regionFlags)
    domain.boundaryTags = boundaryTags
    #domain.polyfile="meshMarin"
    domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"meshRisingBubble"
    #domain.writePoly("meshRisingBubble")
    he = tank_dim[0]/float(4*refinement-1)
    domain.MeshOptions.he = opts.he
    domain.MeshOptions.genMesh=False
    domain.MeshOptions.triangleFlag=0
    triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self,x,t):
        return 0.

class clsvof_init_cond(object):
    def uOfXT(self,x,t):
        xB = 0.5
        yB = 0.5
        rB = 0.25
        zB = 0.0
        # dist to center of bubble
        r = np.sqrt((x[0]-xB)**2 + (x[1]-yB)**2)
        # dist to surface of bubble
        dB = rB - r
        #return dB
        if dB>0:
            return 1.0
        elif dB==0:
            return 0.0
        else:
            return -1.0

#################
# NAVIER STOKES #
#################
# DIRICHLET BCs #
def vel_u_DBC(x,flag):
    if flag==boundaryTags['bottom'] or flag==boundaryTags['top']:
        return lambda x,t: 0.0

def vel_v_DBC(x,flag):
    if flag==boundaryTags['bottom'] or flag==boundaryTags['top']:
        return lambda x,t: 0.0

# ADVECTIVE FLUX #
def vel_u_AFBC(x,flag):
    if not (flag==boundaryTags['bottom'] or flag==boundaryTags['top']):
        return lambda x,t: 0.0

def vel_v_AFBC(x,flag):
    if not (flag==boundaryTags['bottom'] or flag==boundaryTags['top']):
        return lambda x,t: 0.0

# DIFFUSIVE FLUX #
def vel_u_DFBC(x,flag):
    if not (flag==boundaryTags['bottom'] or flag==boundaryTags['top']):
        return lambda x,t: 0.0

def vel_v_DFBC(x,flag):
    if not (flag==boundaryTags['bottom'] or flag==boundaryTags['top']):
        return lambda x,t: 0.0

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
############################################
boundaryConditions = {
    # DIRICHLET BCs #
    'pressure_DBC': lambda x, flag: None,
    'pressure_increment_DBC': lambda x, flag: None,
    'vel_u_DBC': vel_u_DBC,
    'vel_v_DBC': vel_v_DBC,
    'clsvof_DBC': lambda x, flag: None,
    # ADVECTIVE FLUX BCs #
    'pressure_AFBC': lambda x, flag: lambda x, t: 0.0,
    'pressure_increment_AFBC': lambda x, flag: lambda x, t: 0.0,
    'vel_u_AFBC': vel_u_AFBC,
    'vel_v_AFBC': vel_v_AFBC,
    'clsvof_AFBC': lambda x, flag: lambda x, t: 0.0,
    # DIFFUSIVE FLUX BCs #
    'pressure_increment_DFBC': lambda x, flag: lambda x, t: 0.0,
    'vel_u_DFBC': vel_v_DFBC,
    'vel_v_DFBC': vel_v_DFBC,
    'clsvof_DFBC': lambda x, flag: None}

myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.domain = domain
myTpFlowProblem.outputStepping.final_time = opts.final_time
myTpFlowProblem.outputStepping.dt_output = opts.dt_output
myTpFlowProblem.outputStepping.systemStepExact = True

myTpFlowProblem.SystemPhysics.setDefaults()
myTpFlowProblem.SystemPhysics.useDefaultModels(flowModel=opts.ns_model,interfaceModel=1)

myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=zero()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=zero()
myTpFlowProblem.SystemPhysics.modelDict['clsvof'].p.initialConditions['clsvof'] = clsvof_init_cond()
myTpFlowProblem.SystemPhysics.modelDict['pressure'].p.initialConditions['p'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInc'].p.initialConditions['pInc'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInit'].p.initialConditions['pInit'] = zero()

myTpFlowProblem.SystemPhysics.boundaryConditions = boundaryConditions
myTpFlowProblem.SystemPhysics.useBoundaryConditionsModule = False
myTpFlowProblem.SystemPhysics.gravity = [0.0, -0.98, 0.0]
physical_parameters = myTpFlowProblem.SystemPhysics
if opts.test_case==1:
    physical_parameters.rho_0 = 1000.0
    physical_parameters.nu_0 = 10.0/physical_parameters['rho_0']
    physical_parameters.rho_1 = 100.0
    physical_parameters.nu_1 = 1.0/physical_parameters['rho_1']
    physical_parameters.surf_tension_coeff = 24.5
else: #test_case=2
    physical_parameters.rho_0 = 1000.0
    physical_parameters.nu_0 = 10.0/physical_parameters.rho_0
    physical_parameters.rho_1 = 1.0
    physical_parameters.nu_1 = 0.1/physical_parameters.rho_1
    physical_parameters['surf_tension_coeff'] = 1.96

myTpFlowProblem.SystemNumerics.cfl=opts.cfl
myTpFlowProblem.SystemNumerics.useSuperlu=True

m = myTpFlowProblem.SystemPhysics.modelDict
m['flow'].p.coefficients.useVF=1.0
m['flow'].p.coefficients.eb_penalty_constant = 1e6
m['flow'].n.ShockCapturingOptions.shockCapturingFactor = 0.5
m['flow'].p.coefficients.ARTIFICIAL_VISCOSITY = opts.ARTIFICIAL_VISCOSITY
m['clsvof'].p.coefficients.disc_ICs = True
m['clsvof'].p.coefficients.computeMetricsForBubble = True



