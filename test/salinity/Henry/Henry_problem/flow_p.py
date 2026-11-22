from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
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
m_per_s_by_m_per_d = 1.1574074e-5
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional

#base 
permeability1  = (1.2410491586180303e-03)*viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS1        = 0.4   #-
thetaR1        = 0.05   #-
mvg_alpha1     = 8.0   #1/m
mvg_n1         = 2.4
mvg_m1         = 1.0 - 1.0/mvg_n1
dimensionless_conductivity1  = (timeScale*density*gravity*permeability1/(viscosity*lengthScale))

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
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1]#m/d?
    else:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1,dimensionless_conductivity1]#m/d? 
        
galerkin = False   
#useSeepageFace = True

#def getSeepageFace(flag):
#    if useSeepageFace:
#        if flag == boundaryTags['drain']:
#            return 1
#        else:
#            return 0
#    else:
#        return 0
        
        
        
        
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
                                     STABILIZATION_TYPE=2, #0,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX= False ,
                                     FCT=False ,#True,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)#,

#
#galerkin = False

# Density Relation
rho_fw =  1000.0 # density of freshwater
rho_sw = 1025.0 # 1026.0 density of saltwater
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0# .0357 # saline mass fraction, saltwater
epsilon = (rho_sw - rho_fw)/rho_fw 
drho_dmf =epsilon*rho_fw #700. #epsilon*rho_fw # density change with mass fraction
q_in=-6.6e-5
# Dynamic Viscosity Relation
nu = 1.0e-3


# class hydrostaticIC:
#     def __init__(self, val=0.3):
#         self.val = val
#     def uOfXT(self, x, t):
#         # Adjusting from hydraulic head to pressure head
#         return (L[1] - x[1]) * (1.0 + epsilon * self.val)
#     def uOfX(self, x, t):
#         # Adjusting from hydraulic head to pressure head
#         return (L[1] - x[1]) * (1.0 + epsilon * self.val)

class hydrostaticIC:
    #def __init__(self,val = 0.3):
        #self.val = val
    def uOfXT(self,x,t):
        return  0+ x[1]*dimensionless_gravity[1]*dimensionless_density # + (L[1]-x[1])*(1.0+epsilon*self.val) #x[1]
    #def uOfX(self,x,t):
    #    return  x[1] + (L[1]-x[1])*(1.0+epsilon*self.val)
def getHenryheadDirichletBCs(x,tag):
    #pass
   if x[0]<1.e-8:
       return lambda x,t: x[1] +  (L[1]-x[1])*(1.0+epsilon*mf_sw) #mf_sw left boundary inflow
   else: 
       pass

# Flux Conditions 

def getFlowDiffusiveFluxBCs(x,tag):
    if x[0]<1.e-8:
        return lambda x,t: rho_fw*q_in #6.6e-5/2. 
    elif ((x[0]>=1.e-8 and abs(x[0]-L[0]) > 1.e-8) and (x[1] < 1.e-8 or abs(x[1]-L[1])<1.e-8)):
        return lambda x,t: 0.0
    else:
        pass


def getFlowAdvectiveFluxBCs(x,tag):
    if ((x[0]>= 1.e-8 and abs(x[0]-L[0])>1.e-8) and (x[1] < 1.e-8 or abs(x[1]- L[1])<1.e-8)):
        return lambda x,t: 0.0
    else:
        pass


fluxBoundaryConditions = {0:'setFlow'}
#initialConditions={0:hydrostaticIC(mf_fw)}
initialConditions={0:hydrostaticIC()}
dirichletConditions = {0: getHenryheadDirichletBCs}
advectiveFluxBoundaryConditions =  {0:getFlowDiffusiveFluxBCs}
diffusiveFluxBoundaryConditions = {0:{0:getFlowDiffusiveFluxBCs}}

#T = 0.0003/timeScale
