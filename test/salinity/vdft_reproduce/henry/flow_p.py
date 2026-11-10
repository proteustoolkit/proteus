from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
import numpy as np
try:
    from .domain_henry import *
except:
    from domain_henry import *
nd = 2

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 1000 # 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0 #density*gravity*4.524e-10
# Simulation runs in SI: meters, seconds, kg.


Kx            = 0.01 #0.01244
Kz            = Kx
thetaS1       = 0.35   # porosity
thetaR1       = 0.0
mvg_alpha1     = 8.0   #1/m
mvg_n1         = 2.4
mvg_m1         = 1.0 - 1.0/mvg_n1
dimensionless_conductivity_x = Kx
dimensionless_conductivity_z = Kz

#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = np.array([0.0,#
                                        -1.0,
                                        0.0])
#dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,2),'d')

for i in range(nMediaTypes+1):
    if i==1:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity_x,dimensionless_conductivity_z]
    else:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity_x,dimensionless_conductivity_z]

galerkin = False
#useSeepageFace = True
LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=dimensionless_gravity,
                                     density=dimensionless_density,
                                     beta=0.0,
                                     density_model=1,
                                     #beta_c = 0.025,
                                     diagonal_conductivity=True,
                                     #DENSITY_MODEL = 1,
                                     #density_coupling = 1,
                                     STABILIZATION_TYPE=2,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX= False ,
                                     FCT=True, FCT_BND_DIFF=True, FCT_BND_KRW=True,  # Barua edge-flux runs: his FCT boundary block (both off by default since #173)
                                     #True,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)#,
                                     #getSeepageFace=getSeepageFace,
                                     #rho_0 = dimensionless_density,
                                     #storativity = 0.0)

#G= [1.8, 1.2,1]
rho_fw =1000.0
galerkin = False

# Richards takes bc_flux as the outward normal total flux.
# On the left boundary, inflow into the domain is therefore negative.
q_in=  -6.6e-5
#mf_sw = 0.357
boundaryTags = domain.boundaryTags
# class hydrostaticIC:
#     #def __init__(self,val = 0.3):
#         #self.val = val
#     def uOfXT(self,x,t):
#         return  (L[1]-x[1])
#     def uOfX(self,x,t):
#        return  (L[1]-x[1])



# def getHenryheadDirichletBCs(x,tag):
#    if L[0]-x[0] <1.e-8:
#        return lambda x,t: (L[1]-x[1])
#    else:
#        pass

class hydrostaticIC:
    def uOfXT(self,x,t):
        # Freshwater hydrostatic head, consistent with the transport IC u=0.
        # The right Dirichlet BC keeps imposing saltwater hydrostatic, so the
        # salt wedge develops naturally from there.
        return (L[1]-x[1])
    #def uOfX(self,x,t):
    #    return  x[1] + (L[1]-x[1])*(1.0+epsilon*self.val)

rho_fw =  1000.0 # density of freshwater
rho_sw = 1025.0 # 1026.0 density of saltwater
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0# .0357 # saline mass fraction, saltwater
epsilon = (rho_sw - rho_fw)/rho_fw
drho_dmf =epsilon*rho_fw #700. #epsilon*rho_fw # density change with mass fraction



def getHenryheadDirichletBCs(x,tag):
    if tag == boundaryTags['right']:
        return lambda x,t: (rho_sw/rho_fw)*(L[1]-x[1])
    else:
        pass

dirichletConditions = {0:getHenryheadDirichletBCs}
initialConditions  = {0:hydrostaticIC()}

def getFlowAdvectiveFluxBCs(x,tag):
    if tag == boundaryTags['left']:
        return lambda x,t: q_in
    if tag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.0
    else:
        pass

fluxBoundaryConditions = {0:'setFlow'}
advectiveFluxBoundaryConditions =  {0:getFlowAdvectiveFluxBCs}
diffusiveFluxBoundaryConditions = {0:{}}
