from proteus import *
from proteus.default_p import *
from miscible_displacement import *

r"""
advective-dispersive transport in heterogeneous domain

   \pd{\theta c}{t} + \deld \left[\vec q c - \theta\ten{D}\nabla c\right] = s_c(x,t) 

   where \ten{D} is the dispersion tensor
"""

initialConditions = initialConditions_trans

dirichletConditions = {0:concentration_bc}

coefficients = STC.GroundwaterTransportCoefficients(1,
                                                    omega_types=omega_types,
                                                    alpha_L_types=alpha_L_types,
                                                    alpha_T_types=alpha_T_types,
                                                    d = d_mol,
                                                    meModelId=1,
                                                    flowModelId=0)
   


fluxBoundaryConditions = {0:'outFlow'}
advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{}}

