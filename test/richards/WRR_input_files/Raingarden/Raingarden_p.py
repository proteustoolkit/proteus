from proteus import *
from proteus.default_p import *
from proteus.kernels import Richards
from proteus import Context
from domain_rg import *

# Scheme selection.  The defaults are the FCT scheme; the other two are
#   Galerkin              -C "STABILIZATION_TYPE=0 FCT=False"
#   entropy viscosity     -C "STABILIZATION_TYPE=2 FCT=False"
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0: Galerkin (no stabilization), 2: entropy viscosity"),
    ("FCT", True, "apply the flux-corrected-transport limiter"),
], mutable=True)

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0 #density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional

#Mualem-van Genuchten hydraulic parameters (capillary-barrier / anisotropic rain garden).
#Pore-connectivity l=0.5 is hardwired into the VGM k_rw closure (richards/psk_models.h:54,
#k_rw = sqrt(Se)*(...)^2), so it is not passed in here.
#Nominal layer thicknesses from the table (root 0.50 m, storage 0.70 m) are NOT imposed --
#the existing domain_rg.py extents stand in for the layers.

#Root zone -- fine sand
Ks_h1          = 5.0    #m/d  horizontal
Ks_v1          = 2.5    #m/d  vertical  (Ks_h/Ks_v = 2)
thetaS1        = 0.41   #-
thetaR1        = 0.045  #-
mvg_alpha1     = 2.0    #1/m
mvg_n1         = 1.8
mvg_m1         = 1.0 - 1.0/mvg_n1
permeability1_h = Ks_h1*viscosity/(gravity*density)  #m^2
permeability1_v = Ks_v1*viscosity/(gravity*density)  #m^2
dimensionless_conductivity1_h = (timeScale*density*gravity*permeability1_h/(viscosity*lengthScale))
dimensionless_conductivity1_v = (timeScale*density*gravity*permeability1_v/(viscosity*lengthScale))

#Storage zone -- coarse sand/gravel
Ks_h2          = 80.0   #m/d  horizontal
Ks_v2          = 20.0   #m/d  vertical  (Ks_h/Ks_v = 4)
thetaS2        = 0.36   #-
thetaR2        = 0.025  #-
mvg_alpha2     = 15.0   #1/m
mvg_n2         = 3.0
mvg_m2         = 1.0 - 1.0/mvg_n2
permeability2_h = Ks_h2*viscosity/(gravity*density)  #m^2
permeability2_v = Ks_v2*viscosity/(gravity*density)  #m^2
dimensionless_conductivity2_h = (timeScale*density*gravity*permeability2_h/(viscosity*lengthScale))
dimensionless_conductivity2_v = (timeScale*density*gravity*permeability2_v/(viscosity*lengthScale))

#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        -1.0,
                                        0.0])
#dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 2
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,2),'d')

#diagonal_conductivity=True -> KsTypes[i,:] are the diagonal entries [Kxx,Kyy],
#i.e. [K_s,h, K_s,v] since gravity acts along -y.
for i in range(nMediaTypes+1):
    if i==1:
        alphaVGtypes[i] = mvg_alpha1
        nVGtypes[i]     = mvg_n1
        thetaStypes[i]  = thetaS1
        thetaRtypes[i]  = thetaR1
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity1_h,dimensionless_conductivity1_v]#m/d
    elif i==2:
        alphaVGtypes[i] = mvg_alpha2
        nVGtypes[i]     = mvg_n2
        thetaStypes[i]  = thetaS2
        thetaRtypes[i]  = thetaR2
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity2_h,dimensionless_conductivity2_v]#m/d

galerkin = False
useSeepageFace = True

def getSeepageFace(flag):
    if useSeepageFace:
        if flag == boundaryTags['drain']:
            return 1
        else:
            return 0
    else:
        return 0




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
                                     penalty_constant=1.0e2,
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX= False ,
                                     FCT=ct.FCT,
                                     num_fct_iter=0,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False,
                                     getSeepageFace=getSeepageFace)

#G= [1.8, 1.2,1]

galerkin = False

G=[3.0,5.0,1.0]

pondingPressure= 0.1
def getDBC_2D_Richards_Shock(x,flag):
    if x[1] == L[1]:
        if (x[0] >= L[0]/3.0 and
            x[0] <= 2.0*L[0]/3.0):
            return lambda x,t: pondingPressure
    if x[1] == 0.0:
        return None #lambda x,t: 0.0
    if (x[0] == 0.0 or
        x[0] == L[0]):
        return None #lambda x,t: x[1]*dimensionless_gravity[1]*dimensionless_density
   # if flag=="drain":
   #     return lambda x,t: 0.0

dirichletConditions = {0:getDBC_2D_Richards_Shock}

#Uniform IC, matching the HYDRUS run (Domain.dat column h is -5 m at every node).
initialPressureHead = -5.0

class ShockIC_2D_Richards:
    def uOfXT(self,x,t):
        bc=getDBC_2D_Richards_Shock(x,0)
        if bc != None:
            return bc(x,t)
        else:
            return initialPressureHead

initialConditions  = {0:ShockIC_2D_Richards()}

fluxBoundaryConditions = {0:'noFlow'}

def getFBC_2D_Richards_Shock(x,flag):
    if x[1] == G[1]:
        if (x[0] < G[0]/3.0 or
            x[0] > 2.0*G[0]/3.0):
            return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFBC_2D_Richards_Shock}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_2D_Richards_Shock}}
