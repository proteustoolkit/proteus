from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
import numpy as np

nd = 2

#L= []

#L=[3.0,5.0,1.0]
# parallel partitioning info #
from proteus import MeshTools
partitioningType = MeshTools.MeshParallelPartitioningTypes.node
L=(.530,.26,1)#.26 
nd = 2

nnx =107#213#107
nny =53#105#53#53# 105

#unstructured=ct.unstructured #True for tetgen, false for tet or hex from rectangular grid


domain = Domain.RectangularDomain(L=[L[0],L[1]],name="gwvd_gc_domain",units="m")
polyfile="gwvd_gc_domain_2d"
domain.writePoly(polyfile)
triangleOptions = "q30Dena0.00001"



domain.MeshOptions.nnx = nnx
domain.MeshOptions.nny = nny
domain.MeshOptions.triangleFlag=0

regularGrid=False

    
#domain.polyfile="bio2d_n_coarse"
#domain.polyfile="ImageToStl.com_fflower-tc_n"
analyticalSolution = None
    
viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0 #density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional

#base 
permeability1  = (1.2410491586180303e-09)*viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS1        = 0.4   #-
thetaR1        = 0.05   #-
mvg_alpha1     = 8.0   #1/m
mvg_n1         = 2.4
mvg_m1         = 1.0 - 1.0/mvg_n1
dimensionless_conductivity1  = (timeScale*density*gravity*permeability1/(viscosity*lengthScale))

#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = np.array([0.0,
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

#recoeffs= coefficients
                                     #getSeepageFace=getSeepageFace)
#galerkin = False

#coefficients = ConservativeHeadRichardsMualemVanGenuchten(hydraulicConductivity=dimensionless_conductivity,
#                                                          gravity=dimensionless_gravity,
#                                                          density=dimensionless_density,
#                                                          thetaS=thetaS,
#                                                          thetaR=thetaR,
#                                                          alpha= dimensionless_alpha,
#                                                          n = mvg_n,
#                                                          m = mvg_m,
#                                                          beta = beta)


#coefficients = ConservativeHeadRichardsMualemVanGenuchten(hydraulicConductivity=dimensionless_conductivity,
#                                                          gravity=dimensionless_gravity,
#                                                          density=dimensionless_density,
#                                                          thetaS=thetaStypes,
#                                                          thetaR=thetaRtypes,
#                                                          alpha= alphaVGtypes,
#                                                          n = mvg_n,
#                                                          m = mvg_m,
#                                                          beta = beta)#

#G= [1.8, 1.2,1]
#domain.writePLY('CCS')
galerkin = False

#G=[2.799999952316284,1.4950000047683716]
L=[.530,.26,1]
# Density Relation
rho_fw =  998.2#1000 # density of freshwater
rho_sw =1026.10 #  density of saltwater- corresponds to conc=37.1kg/m3
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0# .0357 # saline mass fraction, saltwater
epsilon =(rho_sw - rho_fw)/rho_fw 
drho_dmf =epsilon*rho_fw #700. #epsilon*rho_fw # density change with mass fraction

#pondingPressure= 0.5
#For SS-1: 26.7 cm, SS-2: 26.55 cm, SS-3: 26.2 cm                                                                
h_ss1=.267 # Head for intruding steady state
h_ss3=.2655 # Head for intermediate steady state
h_ss2=.262 # Head for receding steady state
h_saltwater=.255 #
def head_ramp_intruding(x,t):
    if (t<=5000):
        return h_ss1
    elif (t>5000):
        return h_ss2 #h_ss2
 
def head_ramp_receding(x,t):
    if (t<=11000):
        return h_ss2
    elif (t>11000):
        return h_ss2 #h_ss2
# Dirichlet
# def getHeadDirichletBCs_SS(x,tag): # Need to fix this -- should be h_s=25.5 cm at top
#     if (abs(x[0])<1.e-8 and x[1]<=0.255):
#         #return lambda x,t: x[1] + (1.0+epsilon*mf_sw)*(h_saltwater-x[1])
#         return lambda x,t: h_saltwater + epsilon*(h_saltwater-x[1])
#     elif (abs(x[0]-L[0])< 1.e-8):
#         return lambda x,t: h_ss1
#     else:
#         pass

def getHeadDirichletBCs_SS(x, tag):
    if abs(x[0]) < 1.e-8 and x[1] <= 0.255:  # Left boundary (Saltwater region)
        # Apply hydrostatic adjustment for saltwater region
        return lambda x, t: h_saltwater + epsilon * (h_saltwater - x[1])
    elif abs(x[0] - L[0]) < 1.e-8:  # Right boundary (Freshwater region)
        # Set constant head for the freshwater side at steady-state level SS1
        return lambda x, t: h_ss1
    else:
        # Return None for areas without Dirichlet boundary conditions
        return None
    
def getHeadDirichletBCs_Receding(x,tag):                                                                                                
    if (abs(x[0])<1.e-8 and x[1]<=0.255):                                                                                                             
        return lambda x,t: x[1] + (1.0+epsilon*mf_sw)*(h_saltwater-x[1])
    elif (abs(x[0]-L[0])< 1.e-8):
        return lambda x,t: head_ramp_receding(x,t)                                                                                                                                   
    else:
        pass

def getHeadDirichletBCs_Intruding(x,tag): # Need to fix this -- should be h_s=25.5 cm at top                                                
    if (abs(x[0])<1.e-8 and x[1]<=0.255):                                                                                                            
        return lambda x,t: x[1] + (1.0+epsilon*mf_sw)*(.255-x[1])
    elif (abs(x[0]-L[0])< 1.e-8):
        return lambda x,t: head_ramp_intruding(x,t)                                                                                                                                   
    else:
        pass


dirichletConditions = {0:getHeadDirichletBCs_SS}


# class hydrostaticIC:
#     def __init__(self,val = 0.3):
#         self.val = val
#     def uOfXT(self,x,t):
#         return  x[1] + (L[1]-x[1])*(1.0+epsilon*self.val) #x[1]
#     def uOfX(self,x,t):
#         return  x[1] + (L[1]-x[1])*(1.0+epsilon*self.val)
class hydrostaticIC:
    def __init__(self, h_saltwater=0.255, h_freshwater=0.267, epsilon=0.028):
        """
        Initializes the hydrostatic initial condition.
        :param h_saltwater: Saltwater head (m) at the left boundary.
        :param h_freshwater: Freshwater head (m) at the right boundary.
        :param epsilon: Density ratio adjustment between saltwater and freshwater.
        """
        self.h_saltwater = h_saltwater  # Saltwater head
        self.h_freshwater = h_freshwater  # Freshwater head
        self.epsilon = epsilon  # Density adjustment factor

    def uOfXT(self, x, t):
        """
        Computes the hydraulic head or hydrostatic pressure based on the region.
        :param x: [x, y] coordinate of the point.
        :param t: Time (not used for initial conditions).
        :return: Initial hydraulic head or hydrostatic pressure.
        """
        if x[1] <= self.h_saltwater:  # Saltwater region (below interface)
            # Saltwater hydrostatic pressure
            return self.h_saltwater + self.epsilon * (self.h_saltwater - x[1])
        else:  # Freshwater region (above interface)
            # Freshwater hydraulic head
            return self.h_freshwater - x[1]

    def uOfX(self, x, t):
        """
        Wrapper for uOfXT to compute values without time dependency.
        """
        return self.uOfXT(x, t)

# class hydrostaticIC:
#     def __init__(self, val=0.3, h_saltwater=0.255):
#         self.val = val
#         self.h_saltwater = h_saltwater

#     def uOfXT(self, x, t):
#         if x[1] <= self.h_saltwater:  # Saltwater region
#             return self.h_saltwater + (self.h_saltwater - x[1]) * (1.0 + epsilon * self.val)
#         else:  # Freshwater region
#             return x[1]  # Adjust based on the freshwater condition (e.g., head or elevation)

#     def uOfX(self, x, t):
#         return self.uOfXT(x, t)
    
# class ShockIC_2D_Richards:
#     def uOfXT(self,x,t):
#         bc=getDBC_2D_Richards_Shock(x,0)
#         if bc != None:
#             return bc(x,t)
#         else:
#             return 0.0 #x[1]*dimensionless_gravity[1]*dimensionless_density

initialConditions  = {0:hydrostaticIC()}

fluxBoundaryConditions = {0:'noFlow'}

def getFlowDiffusiveFluxBCs(x,tag): 
    if ((x[0]>=1.e-8 and abs(x[0]-L[0]) > 1.e-8) and (x[1] < 1.e-8 or abs(x[1]-L[1])<1.e-8)):
        return lambda x,t: 0.0
    elif (abs(x[0])<1.e-8 and x[1]>0.255):
        return lambda x,t: 0.0
    else:
        pass

def getFlowAdvectiveFluxBCs(x,tag):
    if ((x[0]>= 1.e-8 and abs(x[0]-L[0])>1.e-8) and (x[1] < 1.e-8 or abs(x[1]- L[1])<1.e-8)):
        return lambda x,t: 0.0
    else:
        pass

advectiveFluxBoundaryConditions =  {0:getFlowDiffusiveFluxBCs}

diffusiveFluxBoundaryConditions = {0:{0:getFlowDiffusiveFluxBCs}}

#T = 0.0003/timeScale
