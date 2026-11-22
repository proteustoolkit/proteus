from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
from proteus import Context

# Scheme selection.  The defaults are the FCT scheme; the other one is
#   entropy viscosity     -C "STABILIZATION_TYPE=2 FCT=False"
# test_richards.py drives both through Context.contextOptionsString.  Galerkin
# is not run for this benchmark.
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0: Galerkin (no stabilization), 2: entropy viscosity"),
    ("FCT", True, "apply the flux-corrected-transport limiter"),
], mutable=True)

nd = 1

#20 m column, x[0] measured upward from the bottom (HYDRUS z = x[0] - 20)
L=(20.0,1.0,1.0)

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0     #same for every scheme, so the comparison is controlled
m_per_s_by_m_per_d = 1.1574074e-5
#HYDRUS water flow parameters: Ks = 0.297 m/hour = 7.128 m/d
permeability  = (7.128*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
thetaS        = 0.43    #- (HYDRUS Qs)
thetaR        = 0.045   #- (HYDRUS Qr)
mvg_alpha     = 14.5    #1/m
mvg_n         = 2.68
mvg_m         = 1.0 - 1.0/mvg_n
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional
dimensionless_conductivity  = (timeScale*density*gravity*permeability/(viscosity*lengthScale))/m_per_s_by_m_per_d
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([-1.0,
                                       0.0,
                                       0.0])
dimensionless_alpha    = mvg_alpha*lengthScale
satRichards = False
optRichards = True
nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,1),'d')

for i in range(nMediaTypes+1):
    alphaVGtypes[i] = mvg_alpha
    nVGtypes[i]     = mvg_n
    thetaStypes[i]  = thetaS
    thetaRtypes[i]  = thetaR
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i,:]    = [dimensionless_conductivity]#,dimensionless_conductivity,dimensionless_conductivity]#m/d?

useSeepageFace = True
galerkin=False

# if galerkin:
#     stabilization_type=0
# else:
#     stabilization_type=1

LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=dimensionless_gravity,
                                     density=dimensionless_density,
                                     beta=beta,
                                     diagonal_conductivity=True,
                                     # stiff weak Dirichlet: the ponding BC at the top face
                                     # is only imposed to O(1/penalty), and 100 leaves it
                                     # visibly short of the imposed head on this mesh
                                     penalty_constant=1.0e4,
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=ct.FCT,
                                     MONOLITHIC=False,
                                     num_fct_iter=1,
                                         # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)


#pondingPressure=-0.1#-0.1
#bottomPressure = -0.2#0.0
pondingPressure= 0.0    #zero ponding at the surface, as in the HYDRUS run
bottomPressure = 0.0    #water table at the bottom node, as in the HYDRUS profiles
#pondingSaturation = 0.9
#waterTableSaturation = 0.9
#initialSaturation = 0.01
#pondingPressure=-0.1
# if satRichards:
#     def getDBC_Richards_Shock(x,flag):
#         if x[0] == L[0]:
#             return lambda x,t: pondingSaturation
#         if x[0] == 0.0:
#             return lambda x,t: waterTableSaturation
#else:
def getDBC_Richards_Shock(x,flag):
    if x[0] == L[0]:
        return lambda x,t: pondingPressure
    if x[0] == 0.0:
        return lambda x,t: bottomPressure
   
dirichletConditions = {0:getDBC_Richards_Shock}

# if satRichards:
#     class ShockIC_Richards:
#         def uOfXT(self,x,t):
#             f = getDBC_Richards_Shock(x,0)
#             if f:
#                 return f(x,t)
#             return initialSaturation
# else:
class ShockIC_Richards:
    def uOfXT(self,x,t):
        #HYDRUS initial profile: hydrostatic about the water table at x[0]=0,
        #psi = -x[0], i.e. psi = 0 at the bottom and psi = -L[0] at the surface
        return bottomPressure + x[0]*dimensionless_gravity[0]*dimensionless_density

initialConditions  = {0:ShockIC_Richards()}

fluxBoundaryConditions = {0:'outFlow'}

def flux(x,flag):
    return None
#    if x[0] == L[0]:
#        return lambda x,t: 0.0
#    if x[0] == 0.0:
#        return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:flux}

diffusiveFluxBoundaryConditions = {0:{}}

T = 48.0/24.0/timeScale   #48 hours, in days; column wets through at ~26 h
#T = 0.35/timeScale

#global water-balance diagnostic; writes mass_balance.txt, serial runs only
#import mass_balance_hook
#mass_balance_hook.attach(coefficients, outfile='mass_balance.txt')
