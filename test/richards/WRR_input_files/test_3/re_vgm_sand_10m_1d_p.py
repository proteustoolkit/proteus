from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
from proteus import Context

# Scheme selection.  The defaults are the FCT scheme; the other one is
#   entropy viscosity     -C "STABILIZATION_TYPE=2 FCT=False"
# test_richards.py drives both through Context.contextOptionsString.  Galerkin
# is not run for this benchmark: on the dz = 50 cm grid it undershoots to
# psi = -10.87 m against an initial condition of -7.5 m.
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0: Galerkin (no stabilization), 2: entropy viscosity"),
    ("FCT", True, "apply the flux-corrected-transport limiter"),
    ("fct_type", "semi-implicit", "FCT=True: 'semi-implicit' (WRR) or 'implicit' (JNMA)"),
    ("nn", 0, "nodes in the column; 0 = the benchmark grid"),
    ("nDT", 0, "fixed time steps on [0, T]; 0 = the benchmark's 1000"),
    ("lscheme_L", 0.0, "L-scheme constant, L >= sup theta' (needed for nl_solver lscheme/hybrid)"),
    ("nl_solver", "newton", "newton | lscheme (JNMA defect correction) | hybrid (Newton, L-scheme retry)"),
    ("ls_its", 1, "hybrid: L-scheme iterations replacing a rejected Newton step"),
    ("ls_trigger", "residual", "hybrid: residual (no decrease) | theta (also Theta_k > theta_max)"),
    ("theta_max", 0.5, "hybrid, ls_trigger=theta: contraction threshold"),
    ("fct_fallback", True, "implicit FCT: low-order predictor when the Galerkin predictor fails"),
], mutable=True)

# Szymkiewicz (2009), Water Resour. Res. 45, W10403 -- Test Problem 1:
# infiltration into a 5 m deep dry sand column, soil 5 of Table 1.
# The paper works in cm and h; everything here is m and d.
#   depth  500 cm      -> 5.0 m
#   Ks      21 cm/h    -> 5.04 m/d
#   hg       7.2 cm    -> 0.072 m
#   IC      -750 cm    -> -7.5 m    (theta = 0.069)
#   top BC    -7.5 cm  -> -0.075 m  (theta = 0.422)
#   profiles compared after 3 h -> T = 0.125 d

nd = 1

L=(5.0,1.0,1.0)

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0     #same for every scheme, so the comparison is controlled
m_per_s_by_m_per_d = 1.1574074e-5
permeability  = (5.04*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #Ks = 21 cm/h = 5.04 m/d
thetaS        = 0.43   #-
thetaR        = 0.045  #-
# Soil 5 is Brooks-Corey, so the two shape parameters go into the slots the
# van Genuchten names refer to: alpha is read as the inverse entry-pressure
# head 1/p_d and n as the pore-size index lambda (see Richards.Coefficients).
# The conductivity exponent eta = 2.5 + 2/lambda = 5.88 is not given here --
# PSK_type='BC_MUALEM' selects it in psk_models.h.
bc_pd         = 0.072    #m, entry pressure head hg = 7.2 cm
bc_lambda     = 0.592    #-
mvg_alpha     = 1.0/bc_pd  #1/m
mvg_n         = bc_lambda
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
                                     # Brooks-Corey retention with the Mualem kr
                                     # exponent eta = 2.5 + 2/lambda = 5.88, which
                                     # is what Table 1 of the paper tabulates.
                                     # 'BC' would give Burdine, eta = 6.38.
                                     PSK_type='BC_MUALEM',
                                     diagonal_conductivity=True,
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=ct.FCT,
                                     FCT_TYPE=ct.fct_type,
                                     LSCHEME_L=ct.lscheme_L,
                                     NONLINEAR_SOLVER=ct.nl_solver,
                                     LSCHEME_RETRY_ITS=ct.ls_its,
                                     LSCHEME_TRIGGER=ct.ls_trigger,
                                     LSCHEME_THETA_MAX=ct.theta_max,
                                     FCT_PREDICTOR_FALLBACK=ct.fct_fallback,
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
pondingPressure= -0.075 #0.1
bottomPressure = -7.5
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
        f = getDBC_Richards_Shock(x,0)
        if f:
            return f(x,t)
        else:
            #uniformly dry column; the bottom Dirichlet holds this same head
            return bottomPressure
        #     # return bottomPressure + x[0]*dimensionless_gravity[0]*dimensionless_density
        # if x[0] < L[0]:#*0.5:
        #     return bottomPressure + x[0]*dimensionless_gravity[0]*dimensionless_density
        # else:
        #     return pondingPressure

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

#T = 1/24/timeScale
T = 0.125/timeScale   #3 h, the time Fig. 6 / Table 4 of the paper report
