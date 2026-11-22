"""
TwoPhaseFlow
"""
import numpy as np
from proteus import (Domain, Context, Gauges,
                     MeshTools as mt)
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D
from proteus.SpatialTools import Tank3D
from proteus import SpatialTools as st
from proteus.Profiling import logEvent
import proteus.TwoPhaseFlow.TwoPhaseFlowProblem as TpFlow
import proteus.TwoPhaseFlow.utils.Parameters as Parameters
import math
import os

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts= Context.Options([
    ('ns_model',1,"ns_model={0,1} for {rans2p,rans3p}"),
    ("final_time",7.5,"Final time for simulation"),
    ("dt_output",0.1,"Time interval to output solution"),
    ("gauges", True, "Collect data for validation"),
    ("cfl",0.2,"Desired CFL restriction"),
    ("he",0.5,"Max mesh element diameter"),
    ("ARTIFICIAL_VISCOSITY",3,"artificial viscosity"),
    ("genMesh",False,"generate the mesh on the fly")
    ])

#assert opts.ns_model==1, "use ns_model=1 (rans3pf) for this"

# ****************** #
# ***** GAUGES ***** #
# ****************** #
if opts.gauges:
    pressure_gauges = Gauges.PointGauges(gauges=((('p',),
                                                  ((2.389,0.526,0.025), #P1
                                                   (2.389,0.526,0.099), #P3
                                                   (2.414,0.474,0.165), #P5
                                                   (2.487,0.474,0.165))),), #P7
                                         fileName="pressure.csv")
    point_height_gauges = Gauges.PointGauges(gauges=((('phi',),
                                                      ((2.389,0.526,0.025), #P1
                                                       (2.389,0.526,0.099), #P3
                                                       (2.414,0.474,0.165), #P5
                                                       (2.487,0.474,0.165))),), #P7
                                             fileName="point_clsvof.csv")
    height_gauges = Gauges.LineGauges(gauges=((("phi",),
                                               (((2.724, 0.5, 0.0),
                                                 (2.724, 0.5, 1.0)),
                                                ((2.228, 0.5, 0.0),
                                                 (2.228, 0.5, 1.0)),
                                                ((1.732, 0.5, 0.0),
                                                 (1.732, 0.5, 1.0)),
                                                ((0.582, 0.5, 0.0),
                                                 (0.582, 0.5, 1.0)))),),
                                      fileName="height.csv")

# *************************** #
# ***** DOMAIN AND MESH ***** #
# ****************** #******* #
L      = [3.22,1.0,1.0]
box_L  = [0.16,0.4,0.16]
box_xy = [2.39,0.3]
he = opts.he
boundaries=['left','right','bottom','top','front','back','box_left','box_right','box_top','box_front','box_back',]
boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
bt = boundaryTags
holes = [[0.5*box_L[0]+box_xy[0],0.5*box_L[1]+box_xy[1],0.5*box_L[2]]]
vertices=[[0.0,0.0,0.0],#0
          [L[0],0.0,0.0],#1
          [L[0],L[1],0.0],#2
          [0.0,L[1],0.0],#3
          [0.0,0.0,L[2]],#4
          [L[0],0.0,L[2]],#5
          [L[0],L[1],L[2]],#6
          [0.0,L[1],L[2]],#7
          [box_xy[0],box_xy[1],0.0],#8
          [box_xy[0]+box_L[0],box_xy[1],0.0],#9
          [box_xy[0]+box_L[0],box_xy[1]+box_L[1],0.0],#10
          [box_xy[0],box_xy[1]+box_L[1],0.0],#11
          [box_xy[0],box_xy[1],box_L[2]],#12
          [box_xy[0]+box_L[0],box_xy[1],box_L[2]],#13
          [box_xy[0]+box_L[0],box_xy[1]+box_L[1],box_L[2]],#14
          [box_xy[0],box_xy[1]+box_L[1],box_L[2]]]#15
vertexFlags=[boundaryTags['left'],
             boundaryTags['right'],
             boundaryTags['right'],
             boundaryTags['left'],
             boundaryTags['left'],
             boundaryTags['right'],
             boundaryTags['right'],
             boundaryTags['left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left'],
             boundaryTags['box_left']]
facets=[[[0,1,2,3],[8,9,10,11]],
        [[0,1,5,4]],
        [[1,2,6,5]],
        [[2,3,7,6]],
        [[3,0,4,7]],
        [[4,5,6,7]],
        [[8,9,13,12]],
        [[9,10,14,13]],
        [[10,11,15,14]],
        [[11,8,12,15]],
        [[12,13,14,15]]]
facetFlags=[boundaryTags['bottom'],
            boundaryTags['front'],
            boundaryTags['right'],
            boundaryTags['back'],
            boundaryTags['left'],
            boundaryTags['top'],
            boundaryTags['box_front'],
            boundaryTags['box_right'],
            boundaryTags['box_back'],
            boundaryTags['box_left'],
            boundaryTags['box_top']]
regions=[[0.5*L[0],0.5*L[1],0.5*L[2]]]
regionFlags=[0]
domain = Domain.PiecewiseLinearComplexDomain(vertices=vertices,
                                             vertexFlags=vertexFlags,
                                             facets=facets,
                                             facetFlags=facetFlags,
                                             regions = regions,
                                             regionFlags = regionFlags,
                                             holes=holes)
domain.MeshOptions.setParallelPartitioningType('node')
domain.boundaryTags = boundaryTags
#domain.polyfile="meshMarin"
domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"meshMarin"
#domain.writePLY("mesh")
#domain.writeAsymptote("mesh")
domain.MeshOptions.triangleOptions="VApq1.25q12feena%e" % ((he**3)/6.0,)
domain.MeshOptions.he = opts.he
domain.MeshOptions.triangleFlag = 0
domain.MeshOptions.genMesh=opts.genMesh
if opts.genMesh:
    domain.writePoly("meshMarin")

# ****************************** #
# ***** INITIAL CONDITIONS ***** #
# ****************************** #
class zero(object):
    def uOfXT(self,x,t):
        return 0.

disc_ICs=True
class clsvof_init_cond(object):
    def uOfXT(self,x,t):
        waterLine_x = 1.22
        waterLine_z = 0.55
        phi_x = x[0]-waterLine_x
        phi_z = x[2]-waterLine_z
        if disc_ICs:
            if x[0] < waterLine_x and x[2] < waterLine_z:
                return -1.0
            elif x[0] > waterLine_x or x[2] > waterLine_z:
                return 1.0
            else:
                return 0.0
        else:
            if phi_x < 0.0:
                if phi_z < 0.0:
                    return max(phi_x,phi_z)
                else:
                    return phi_z
            else:
                if phi_z < 0.0:
                    return phi_x
                else:
                    return math.sqrt(phi_x**2 + phi_z**2)

# ******************************* #
# ***** BOUNDARY CONDITIONS ***** #
# ******************************* #
non_slip_BCs=True
openTop=True
# DIRICHLET BOUNDARY CONDITIONS #
def vel_u_DBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return lambda  x,t: 0.0

def vel_v_DBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return lambda  x,t: 0.0

def vel_w_DBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return lambda  x,t: 0.0

def pressure_increment_DBC(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

def pressure_DBC(x,flag):
    if flag == boundaryTags['top'] and openTop:
        return lambda x,t: 0.0

def clsvof_DBC(x,flag):
    if openTop and flag == boundaryTags['top']:
        return lambda x,t: 1.0

# ADVECTIVE FLUX BOUNDARY CONDITIONS #
def vel_u_AFBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return None
    elif openTop and flag == boundaryTags['top']:
        return None
    else: #slip everywhere but the box
        return lambda x,t: 0.0

def vel_v_AFBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return None
    elif openTop and flag == boundaryTags['top']:
        return None
    else: #slip everywhere but the box
        return lambda x,t: 0.0

def vel_w_AFBC(x,flag):
    if non_slip_BCs and (flag == boundaryTags['box_left'] or
                         flag == boundaryTags['box_right'] or
                         flag == boundaryTags['box_top'] or
                         flag == boundaryTags['box_front'] or
                         flag == boundaryTags['box_back']):
        return None
    elif openTop and flag == boundaryTags['top']:
        return None
    else: #slip everywhere but the box
        return lambda x,t: 0.0

def pressure_increment_AFBC(x,flag):
    if not (flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

def pressure_AFBC(x,flag):
    if not(flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

def clsvof_AFBC(x,flag):
    if openTop and flag == boundaryTags['top']:
        return None
    else:
        return lambda x,t: 0.0

# DIFFUSIVE FLUX BCs #
def pressure_increment_DFBC(x,flag):
    if not (flag == boundaryTags['top'] and openTop):
        return lambda x,t: 0.0

############################################
# ***** Create myTwoPhaseFlowProblem ***** #
###########################################
initialConditions = {'pressure': zero(),
                     'pressure_increment': zero(),
                     'vel_u': zero(),
                     'vel_v': zero(),
                     'vel_w': zero(),
                     'clsvof': clsvof_init_cond()}
boundaryConditions = {
    # DIRICHLET BCs #
    'pressure_DBC': pressure_DBC,
    'pressure_increment_DBC':  pressure_increment_DBC,
    'vel_u_DBC': vel_u_DBC,
    'vel_v_DBC': vel_v_DBC,
    'vel_w_DBC': vel_w_DBC,
    'clsvof_DBC': clsvof_DBC,
    # ADVECTIVE FLUX BCs #
    'pressure_AFBC': pressure_AFBC,
    'pressure_increment_AFBC': pressure_increment_AFBC,
    'vel_u_AFBC': vel_u_AFBC,
    'vel_v_AFBC': vel_v_AFBC,
    'vel_w_AFBC': vel_w_AFBC,
    'clsvof_AFBC': clsvof_AFBC,
    # DIFFUSIVE FLUX BCs #
    'pressure_increment_DFBC': pressure_increment_DFBC,
    'vel_u_DFBC': lambda x, flag: lambda x,t: 0.,
    'vel_v_DFBC': lambda x, flag: lambda x,t: 0.,
    'vel_w_DFBC': lambda x, flag: lambda x,t: 0.,
    'clsvof_DFBC': lambda x, flag: None}


myTpFlowProblem = TpFlow.TwoPhaseFlowProblem()
myTpFlowProblem.domain=domain

myTpFlowProblem.outputStepping.final_time = opts.final_time
myTpFlowProblem.outputStepping.dt_output = opts.dt_output
myTpFlowProblem.outputStepping.systemStepExact = True

myTpFlowProblem.SystemPhysics.setDefaults()
#myTpFlowProblem.SystemPhysics.useDefaultModels(flowModel=opts.ns_model,interfaceModel=1)
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelCLSVOF,'clsvof')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelRANS3PF,'flow')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressureIncrement,'pressureInc')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressure,'pressure')
myTpFlowProblem.SystemPhysics.addModel(Parameters.ParametersModelPressureInitial,'pressureInit')
myTpFlowProblem.SystemPhysics.gravity = np.array([0.0,0.0,-9.8])

myTpFlowProblem.SystemPhysics.boundaryConditions=boundaryConditions
myTpFlowProblem.SystemPhysics.useBoundaryConditionsModule = False
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['u']=zero()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['v']=zero()
myTpFlowProblem.SystemPhysics.modelDict['flow'].p.initialConditions['w']=zero()
myTpFlowProblem.SystemPhysics.modelDict['clsvof'].p.initialConditions['clsvof'] = clsvof_init_cond()
myTpFlowProblem.SystemPhysics.modelDict['pressure'].p.initialConditions['p'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInc'].p.initialConditions['pInc'] = zero()
myTpFlowProblem.SystemPhysics.modelDict['pressureInit'].p.initialConditions['pInit'] = zero()

m = myTpFlowProblem.SystemPhysics.modelDict
m['clsvof'].p.coefficients.disc_ICs = disc_ICs
m['flow'].p.coefficients.useVF = 1.0
m['flow'].p.coefficients.eb_bc_penalty_constant = 1e6
m['flow'].p.coefficients.ARTIFICIAL_VISCOSITY = opts.ARTIFICIAL_VISCOSITY
if opts.gauges:
    m['pressure'].auxiliaryVariables = [pressure_gauges]

myTpFlowProblem.SystemNumerics.cfl=opts.cfl
myTpFlowProblem.SystemNumerics.useSuperlu=True

