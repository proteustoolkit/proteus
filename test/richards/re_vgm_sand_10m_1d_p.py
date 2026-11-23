from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards

opts= Context.Options([
    ('num','fct',"numerics: num = fct, low-order, galerkin, low-order-galerkin, vms-galerkin, vms-sc-galerkin"),
    ("final_time",0.38,"Final time for simulation in days"),
    ("dt",0.1,"Time step for simulation in days"),
    ("nnx",101,"Number of nodes in x direction"),
    ("nny",1,"Number of nodes in y direction"),
    ("nnz",1,"Number of nodes in z direction"),
    ])

nd = 1
he = 10.0/(opts.nnx-1)
L=(10.0,1.0,1.0)
if opts.nny > 1:
    nd = 2
    L=(10.0,10.0/(opts.nny-1),1.0)
if opts.nnz > 1:
    nd = 3
    L=(10.0,10.0/(opts.nny-1),10.0/(opts.nnz-1))

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
if opts.num ==  'low-order-galerkin':
    beta          = density*gravity*4.524e-10
else:
    beta = 0.0
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
                                    STABILIZATION_TYPE=stabilization_type,
                                    ENTROPY_TYPE=0,
                                    LUMPED_MASS_MATRIX=False,
                                    FCT=FCT,
                                    VMS=VMS,
                                    SC=SC,      
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

pondingPressure= 0.1
bottomPressure = 0.0

def getDBC(x,flag):
    if x[0] == L[0]:
        return lambda x,t: pondingPressure
    if x[0] == 0.0:
        return lambda x,t: bottomPressure

dirichletConditions = {0:getDBC}





def flux(x,flag):
    if x[0] == L[0] or x[0] == 0.0:
        return None
    else:
        return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:flux}

diffusiveFluxBoundaryConditions = {0:{}}

class ShockIC_Richards:
    def uOfXT(self,x,t):
        if x[0] < L[0]:
            return bottomPressure + x[0]*dimensionless_gravity[0]*dimensionless_density
        else:
            return pondingPressure

initialConditions  = {0:ShockIC_Richards()}

T = opts.final_time/timeScale
