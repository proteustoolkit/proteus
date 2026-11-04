from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
from domain_rg import *

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0 #density*gravity*4.524e-10
m_per_s_by_m_per_d = 1.1574074e-5
lengthScale   = 1.0     #m
timeScale     = 1.0     #d #1.0/sqrt(g*lengthScale)
#make non-dimensional

# Storage Zone
permeability3  = (5.0)*viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS3        = 0.4   #-
thetaR3        = 0.05   #-
mvg_alpha3     = 8   #1/m
mvg_n3         = 2.4
mvg_m3         = 1.0 - 1.0/mvg_n3
dimensionless_conductivity3  = (timeScale*density*gravity*permeability3/(viscosity*lengthScale))

#Bioswale Zone
permeability2  = (7.128)*viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS2        = 0.43   #-
thetaR2        = 0.045   #-
mvg_alpha2     = 14.5   #1/m
mvg_n2         = 2.68
mvg_m2         = 1.0 - 1.0/mvg_n2
dimensionless_conductivity2  = (timeScale*density*gravity*permeability2/(viscosity*lengthScale))

#base
permeability1  = (1.06)*viscosity/(gravity*density)  #m^2
#permeability1  = (0.00504)*viscosity/(gravity*density)  #m^2
thetaS1        = 0.41   #-
thetaR1        = 0.065   #-
mvg_alpha1     = 7.5   #1/m
mvg_n1         = 1.89
mvg_m1         = 1.0 - 1.0/mvg_n1
dimensionless_conductivity1  = (timeScale*density*gravity*permeability1/(viscosity*lengthScale))
#pipe
#permeability4  = (8.03*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
#thetaS4        = 0.43   #-
#thetaR4        = 0.045   #-
#mvg_alpha4     = 20   #1/m
#mvg_n4         = 3
#mvg_m4         = 1.0 - 1.0/mvg_n4
#dimensionless_conductivity4  = (timeScale*density*gravity*permeability4/(viscosity*lengthScale))

#print 'Ks',dimensionless_conductivity
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([0.0,
                                        -1.0,
                                        0.0])
#dimensionless_alpha    = mvg_alpha*lengthScale
nMediaTypes  = 3
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
    elif i==2:
        alphaVGtypes[i] = mvg_alpha2
        nVGtypes[i]     = mvg_n2
        thetaStypes[i]  = thetaS2
        thetaRtypes[i]  = thetaR2
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity2,dimensionless_conductivity2]#m/d?

    elif i==3:
        alphaVGtypes[i] = mvg_alpha3
        nVGtypes[i]     = mvg_n3
        thetaStypes[i]  = thetaS3
        thetaRtypes[i]  = thetaR3
        thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
        KsTypes[i,:]    = [dimensionless_conductivity3,dimensionless_conductivity3]#m/d?
    #else:
    #    alphaVGtypes[i] = mvg_alpha4
    #    nVGtypes[i]     = mvg_n4
    #    thetaStypes[i]  = thetaS4
    #    thetaRtypes[i]  = thetaR4
    #    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    #    KsTypes[i,:]    = [dimensionless_conductivity4,dimensionless_conductivity4]#m/d?

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
                                     STABILIZATION_TYPE=0,
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
                                     outputQuantDOFs=False,
                                     getSeepageFace=getSeepageFace)

#G= [1.8, 1.2,1]

galerkin = False

G=[3.0,5.0,1.0]

pondingPressure= 0.5
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

class ShockIC_2D_Richards:
    def uOfXT(self,x,t):
        bc=getDBC_2D_Richards_Shock(x,0)
        if bc != None:
            return bc(x,t)
        else:
            return x[1]*dimensionless_gravity[1]*dimensionless_density

initialConditions  = {0:ShockIC_2D_Richards()}

fluxBoundaryConditions = {0:'noFlow'}

def getFBC_2D_Richards_Shock(x,flag):
    if x[1] == G[1]:
        if (x[0] < G[0]/3.0 or
            x[0] > 2.0*G[0]/3.0):
            return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFBC_2D_Richards_Shock}

diffusiveFluxBoundaryConditions = {0:{0:getFBC_2D_Richards_Shock}}
