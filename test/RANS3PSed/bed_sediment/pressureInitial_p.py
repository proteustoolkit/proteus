from math import *
from proteus import *
from proteus.default_p import *
from proteus.mprans import PresInit
from proteus import Context
from tank import *

ct = Context.get()
name = "pressureInitial"

coefficients=PresInit.Coefficients(nd=nd,
                                   modelIndex=ct.PI_model,
                                   fluidModelIndex=ct.V_model,
                                   pressureModelIndex=ct.P_model)

#pressure increment should be zero on any pressure dirichlet boundaries

#the advectiveFlux should be zero on any no-flow  boundaries


class getIBC_pInit:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class PerturbedSurface_p:
    def __init__(self,waterLevel):
        self.waterLevel=waterLevel
    def uOfXT(self,x,t):
        self.vos = ct.vos_function(x)
        self.rho_a = rho_1#(1.-self.vos)*rho_1 + (self.vos)*ct.rho_s
        self.rho_w = rho_0#(1.-self.vos)*rho_0 + (self.vos)*ct.rho_s
        if signedDistance(x) < 0:
            return -(L[1] - self.waterLevel)*self.rho_a*g[1] - (self.waterLevel - x[1])*self.rho_w*g[1]
        else:
            return -(L[1] - x[1])*self.rho_a*g[1]

initialConditions = {0:PerturbedSurface_p(waterLine_z)} #getIBC_pInit()}

dirichletConditions = {0: lambda x, flag: domain.bc[flag].pInit_dirichlet.init_cython()}
advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].pInit_advective.init_cython()}
diffusiveFluxBoundaryConditions = {0:{0: lambda x, flag: domain.bc[flag].pInit_diffusive.init_cython()}}
