from math import *
from proteus import *
from proteus.default_p import *
from tank import *
import tank_so
from proteus.mprans import PresInc
from proteus import Context

ct = Context.get()


#domain = ctx.domain
#nd = ctx.nd
name = "pressureincrement"

LevelModelType = PresInc.LevelModel
#from ProjectionScheme import PressureIncrement
#coefficients=PressureIncrement(rho_f_min = rho_1,
#                               rho_s_min = rho_s,
#                               nd = nd,
#                               modelIndex=PINC_model,
#                               fluidModelIndex=V_model)
from proteus.mprans import PresInc
coefficients=PresInc.Coefficients(rho_f_min = (1.0-1.0e-8)*rho_1,
                                  rho_s_min = (1.0-1.0e-8)*rho_s,
                                  nd = nd,
                                  modelIndex= ct.DP_model,
                                  fluidModelIndex= ct.V_model,
                                  sedModelIndex= ct.SED_model,
                                  VOF_model= ct.VOF_model,
                                  VOS_model= ct.VOS_model,
                                  fixNullSpace=fixNullSpace_PresInc, 
                                  INTEGRATE_BY_PARTS_DIV_U=ct.INTEGRATE_BY_PARTS_DIV_U_PresInc)


#pressure increment should be zero on any pressure dirichlet boundaries

#the advectiveFlux should be zero on any no-flow  boundaries

class getIBC_phi:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_phi()}

dirichletConditions = {0: lambda x, flag: domain.bc[flag].pInc_dirichlet.init_cython()}
advectiveFluxBoundaryConditions = {0: lambda x, flag: domain.bc[flag].pInc_advective.init_cython()}
diffusiveFluxBoundaryConditions = {0:{0: lambda x, flag: domain.bc[flag].pInc_diffusive.init_cython()}}
