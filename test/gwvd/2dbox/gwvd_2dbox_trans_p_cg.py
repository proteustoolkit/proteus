from proteus import *
from proteus.default_p import *
from proteus.gwvd import GwvdCoefficients
from gwvd_2dbox_cg  import *
import numpy
from math import *


##################################################



analyticalSolution = transportAnalyticalSolution



coefficients =GwvdCoefficients.VDGroundwater_mod2_TransportCoefficients(nctrans,nd=nd,
                                                  omega=omega,
                                                  alpha_L=alpha_L,
                                                  alpha_T=alpha_T,
                                                  d = d_mol,rho_fw=rho_fw,epsilon=epsilon)
   

initialConditions = {0:variableIC(mf_fw,mf_sw)} #constantIC(backgroundConc)}

dirichletConditions ={0:dummyBCs} #{0: getsideDirichletBCs} #  

fluxBoundaryConditions={0: 'noFlow'} # {0:'setFlow'} # {0:'outFlow'}

advectiveFluxBoundaryConditions = {0:getNoFlowBCs}

diffusiveFluxBoundaryConditions = {0:{0:getNoFlowBCs}}


