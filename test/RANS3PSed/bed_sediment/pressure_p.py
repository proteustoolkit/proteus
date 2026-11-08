from math import *
from proteus import *
from proteus.default_p import *
from proteus.mprans import Pres
from proteus import Context
from tank import *

ct = Context.get()
name = "pressure"

LevelModelType = Pres.LevelModel

coefficients=Pres.Coefficients(modelIndex=ct.P_model,
                               fluidModelIndex=ct.V_model,
                               pressureIncrementModelIndex=ct.DP_model,
                               useRotationalForm=True)


class getIBC_p:
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

initialConditions = {0:getIBC_p(waterLine_z)}

dirichletConditions = {0: lambda x, flag: domain.bc[flag].p_dirichlet.init_cython() } # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].pInit_advective.init_cython()}
