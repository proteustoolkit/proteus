from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
import os

nd=3
try:
    from .griffiths_lane_6 import *
except:
    from griffiths_lane_6 import *
he = 4.0
#he*=0.5
#he*=0.5
#he*=0.5
#he*=0.33
#he*=0.5
domain = gl_6_3d(width=he)
boundaryFlags = domain.boundaryFlags
#domain.regionConstraints = [(he**3)/6.0]
domain.regionConstraints = [128.0]
domain.polyfile=os.path.dirname(os.path.abspath(__file__))+"/"+"gl_6_3d"
#domain.writePoly("gl_6_3d")
#domain.writePLY("gl_6_3d")
domain.MeshOptions.genMesh=False#True
triangleOptions="VApfeen"
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
        return lambda x,t: psi0.uOfXT(x,0)#leftHead - x[2] 
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
                                         STABILIZATION_TYPE=0,
                                         FCT=False,
                                         getSeepageFace=getSeepageFace)
