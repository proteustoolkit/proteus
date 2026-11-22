from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
nd = 2

L=(10.0,10.0,1.0)
regularGrid=False
domain = Domain.RectangularDomain(L[:2])
if not regularGrid:
    domain.writePoly('sand2d')
    domain = Domain.PlanarStraightLineGraphDomain('sand2d')

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0#density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
permeability  = (5.04*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
thetaS        = 0.301   #-
thetaR        = 0.093   #-
mvg_alpha     = 5.47    #1/m
mvg_n         = 4.264
mvg_m         = 1.0 - 1.0/mvg_n
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional
dimensionless_conductivity  = (timeScale*density*gravity*permeability/(viscosity*lengthScale))/m_per_s_by_m_per_d
#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        -1.0,
                                        0.0])
dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,2),'d')

for i in range(nMediaTypes+1):
    alphaVGtypes[i] = mvg_alpha
    nVGtypes[i]     = mvg_n
    thetaStypes[i]  = thetaS
    thetaRtypes[i]  = thetaR
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i,:]    = [dimensionless_conductivity,dimensionless_conductivity]#m/d?

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
                                     STABILIZATION_TYPE=2,#0 for galerkin, 2 for Low-order monotone and FCT
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=False,#True,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)
galerkin = False
# coefficients = ConservativeHeadRichardsMualemVanGenuchten(hydraulicConductivity=dimensionless_conductivity,
#                                                           gravity=dimensionless_gravity,
#                                                           density=dimensionless_density,
#                                                           thetaS=thetaS,
#                                                           thetaR=thetaR,
#                                                           alpha= dimensionless_alpha,
#                                                           n = mvg_n,
#                                                           m = mvg_m,
#                                                           beta = beta)

pondingPressure = 0.1
bottomPressure = 0.0

def getDBC_2D_Richards_Shock(x,flag):
    if x[1] == L[1]:
        if (x[0] >= L[0]/3.0 and
            x[0] <= 2.0*L[0]/3.0):
            return lambda x,t: pondingPressure
    if x[1] == 0.0:
        return lambda x,t: 0.0
    if (x[0] == 0.0 or
        x[0] == L[0]):
        return lambda x,t: pondingPressure + x[1]*dimensionless_gravity[1]*dimensionless_density

dirichletConditions = {0:getDBC_2D_Richards_Shock}

class ShockIC_2D_Richards:
    def uOfXT(self,x,t):
        if (x[0] >= L[0]/3.0 and
            x[0] <= 2.0*L[0]/3.0 and
            x[1] == L[1]):
            return pondingPressure
        else:
            return bottomPressure + x[1]*dimensionless_gravity[1]*dimensionless_density

initialConditions  = {0:ShockIC_2D_Richards()}

fluxBoundaryConditions = {0:'noFlow'}

def getFBC_2D_Richards_Shock(x,flag):
   if x[1] == L[1]:
       if (x[0] < L[0]/3.0 or
           x[0] > 2.0*L[0]/3.0):
           return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFBC_2D_Richards_Shock}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_2D_Richards_Shock}}

T = 0.35/timeScale