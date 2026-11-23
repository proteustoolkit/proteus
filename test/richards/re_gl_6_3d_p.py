from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
import os

nd=3
try:
    from .griffiths_lane_6 import *
except:
    from griffiths_lane_6 import *

opts= Context.Options([
    ('num','low-order',"numerics: num = fct, low-order, galerkin, low-order-galerkin, vms-galerkin, vms-sc-galerkin"),
#    ("final_time",0.38,"Final time for simulation in days"),
#    ("dt",0.1,"Time step for simulation in days"),
#    ("nnx",11,"Number of nodes in x direction"),
#    ("nny",1,"Number of nodes in y direction"),
#    ("nnz",1,"Number of nodes in z direction"),
    ("r",3,"refinement factor for mesh"),
    ])

he = 4.0*0.5**opts.r
domain = gl_6_3d(width=he)
boundaryFlags = domain.boundaryFlags
#domain.regionConstraints = [(he**3)/6.0]
domain.regionConstraints = [128.0]
domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"gl_6_3d"
domain.writePoly("gl_6_3d")
#domain.writePLY("gl_6_3d")
domain.MeshOptions.genMesh=True
triangleOptions="VApfeena{0}".format(he)
dimensionless_gravity  = numpy.array([0.0,
                                      0.0,
                                      -1.0])
dimensionless_density  = 1.0
#
#
nMediaTypes  = len(domain.regionLegend)
alphaVGtypes = numpy.zeros((nMediaTypes,),'d')
nVGtypes     = numpy.zeros((nMediaTypes,),'d')
thetaStypes  = numpy.zeros((nMediaTypes,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes,),'d')
KsTypes      = numpy.zeros((nMediaTypes,3),'d')

for i in range(nMediaTypes):
    alphaVGtypes[i] = 5.470
    nVGtypes[i]     = 4.264
    thetaStypes[i]  = 0.301
    thetaRtypes[i]  = 0.308*0.301
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i,:]    = [5.04,5.04,5.04]#m/d?

useSeepageFace = True

leftHead  = 17.1+7.3
rightHead = 7.3
rightHeadInit  = rightHead
leftHeadInit  = leftHead

class SaturatedIC(object):
    def uOfXT(self,x,t):
        #return leftHeadInit - x[2]
        xL = 33.5+leftHeadInit*tan(2.0*pi*18.0/360.0)
        xR = 124.4+33.5
        if (x[0] > xR):
            return rightHeadInit - x[2]
        if (x[0] < xL):
            return leftHeadInit - x[2]
        else:
            return rightHeadInit*(x[0] - xL)/(xR-xL) + leftHeadInit*(xR - x[0])/(xR-xL) - x[2]
  
psi0 = SaturatedIC()
              
initialConditions  = {0:psi0}

def getDBC(x,flag):
    if flag in [boundaryFlags['left'],boundaryFlags['leftTop']]:
        return lambda x,t: psi0.uOfXT(x,0)+10.0#leftHead - x[2] 
    elif flag == boundaryFlags['right']:
        return lambda x,t: psi0.uOfXT(x,0)#rightHead - x[2]
    elif flag == boundaryFlags['rightTop']:
        if useSeepageFace:
            return lambda x,t: 0.0
        else:
            return lambda x,t: psi0.uOfXT(x,0)#rightHead - x[2]
    else:
        return None

dirichletConditions = {0:getDBC}

fluxBoundaryConditions = {0:'mixedFlow'}

def getAFBC(x,flag):
    if flag in [boundaryFlags['left'],
                boundaryFlags['leftTop'],
                boundaryFlags['rightTop'],
                boundaryFlags['right']]:
        return None
    else:
        return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getAFBC}

def getDFBC(x,flag):
    if flag in [boundaryFlags['left'],
                boundaryFlags['leftTop'],
                boundaryFlags['rightTop'],
                boundaryFlags['right']]:
        return None
    else:
        return lambda x,t: 0.0

diffusiveFluxBoundaryConditions = {0:{0:getDFBC}}

def getSeepageFace(flag):
    if useSeepageFace:
        if flag == boundaryFlags['rightTop']:
            return 1
        else:
            return 0
    else:
        return 0
if opts.num == 'fct':
    stabilization_type='EV_Stab'
    FCT=True
    VMS=0.0
    SC=0.0
elif opts.num == 'low-order':
    stabilization_type='EV_Stab'
    FCT=False
    VMS=0.0
    SC=0.0
elif opts.num == 'galerkin':
    stabilization_type='Galerkin'
    FCT=False
    VMS=0.0
    SC=0.0
elif opts.num == 'low-order-galerkin':
    stabilization_type='Galerkin'
    FCT=False
    VMS=0.0
    SC=0.0
elif opts.num == 'vms-galerkin':
    stabilization_type='Galerkin'
    FCT=False
    VMS=1.0
    SC=0.0
elif opts.num == 'vms-sc-galerkin':
    stabilization_type='Galerkin'
    FCT=False
    VMS=1.0
    SC=0.9
else:
    raise Exception("Unknown numerical method: %s" % opts.num)

useOpt = True
if not useOpt:
    coefficients = ConservativeHeadRichardsMualemVanGenuchten(nd,
                                                              KsTypes,
                                                              nVGtypes,
                                                              alphaVGtypes,
                                                              thetaRtypes,
                                                              thetaSRtypes,
                                                              gravity=dimensionless_gravity,
                                                              density=dimensionless_density,
                                                              beta=0.0001,
                                                              diagonal_conductivity=True,
                                                              getSeepageFace=getSeepageFace)
else:
    LevelModelType = Richards.LevelModel
    coefficients = Richards.Coefficients(nd,
                                         KsTypes,
                                         nVGtypes,
                                         alphaVGtypes,
                                         thetaRtypes,
                                         thetaSRtypes,
                                         gravity=dimensionless_gravity,
                                         density=dimensionless_density,
                                         beta=0.0001,
                                         diagonal_conductivity=True,
                                         # FOR EDGE BASED EV
                                        #STABILIZATION_TYPE='Galerkin',#EntropyViscosity',
                                        STABILIZATION_TYPE=stabilization_type,
                                        ENTROPY_TYPE=2,  # logarithmic
                                        LUMPED_MASS_MATRIX=False,
                                        MONOLITHIC=False,
                                        VMS=VMS,
                                        SC=SC,
                                        FCT=FCT,
                                        num_fct_iter=1,
                                        # FOR ENTROPY VISCOSITY
                                        cE=1.0,
                                        uL=0.0,
                                        uR=1.0,
                                        # FOR ARTIFICIAL COMPRESSION
                                        cK=1.0,
                                        # OUTPUT quantDOFs
                                        outputQuantDOFs=False,
                                         getSeepageFace=getSeepageFace)
