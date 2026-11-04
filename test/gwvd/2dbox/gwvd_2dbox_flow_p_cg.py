from proteus import *
from proteus.default_p import *
from proteus.gwvd import GwvdCoefficients
from gwvd_2dbox_cg import *
import numpy
from math import *

"""
Variable Density Heterogenous Flow in a 2D Box.  This verification model
is meant to test the consistent velocity approximation. There are three different
cases for the verification problem.
"""
#########################
# Defining a function for getting consistent velocity with P1 basis on 
# triangular elements
# define some common auxiliary functions for implementation
#########################

# consistentVelocityFlag = 0 #0  -- none, 1 -- Knabner-Frolkovic 96 

def knabnerFrolkovicConsistentVelCorrection_Triangle_P1(nElements,
                                                       c_l2g,     #local to global mapping for concentration fem rep
                                                       c_dof,     #concentration degrees of freedom
                                                       J,         #Jacobian for (affine) element mapping
                                                       Jinv,      #Jacobian inverse
                                                       rho_eval,  #function for (density) \rho(c)
                                                       rhoInt_eval,#function for \int \rho(c) ds
                                                       gravity,   #g
                                                       rhog_consistent,#'consistent' approximation for rho*g
                                                       eps_tol=1.0e-9):#small 
    """
    Knabner-Frolkovic '96 approximation for consistent velocity on
    triangles assuming P^1 approximation for concentration

    Routine just loops through the elements in the mesh and applies
    equations 3.3 -- 3.16 from Knabner-Frolkovic 96

    The output is a nElements x nSpace (2) array holding
      \widetilde{\rho \vec g} =\ten{J}^{-T} . \vec {\hat h} 
    where
      \hat{h} = [\pd{h_{\hat{x}}}{\hat{x}},\pd{h_{\hat{y}}}{\hat{y}}]^{T}

    Note that here the \hat{x},\hat{y} refer to the standard reference
    element coordinates for triangles

       \hat{N}_0 = 1 - \hat{x}  -\hat{y}, \hat{N}_1 = \hat{x}, \hat{N}_2 = \hat{y}

    and that \ten{J} is the transpose of the Jacobian, J, from Knabner and Frolkovic 96

      
    Input:
      c_l2g       --- local to global mapping for concentration fem rep: nElements x nDOF_element (3)
      c_dof       --- concentration degrees of freedom: nDOFs_global
      J           --- Jacobian for (affine) element mapping: nElements x nQuadraturePoints_element x nSpace x nSpace  
      Jinv        --- Jacobian inverse: nElements x nQuadraturePoints_element x nSpace x nSpace
      rho_eval    --- function for (density) \rho(c)
      rhoInt_eval --- function for \int \rho(c) ds
      gravity     --- g
                                                       
    """
    #check some dimensions
    nSpace = 2
    assert c_l2g.shape[0] == nElements; assert c_l2g.shape[1] == nSpace+1
    assert J.shape[0]     == nElements; assert J.shape[2]     == nSpace; assert J.shape[3]    == nSpace
    assert Jinv.shape[0]  == nElements; assert Jinv.shape[2]  == nSpace; assert Jinv.shape[3] == nSpace
    assert gravity.shape[0] == nSpace
    assert rhog_consistent.shape[0] == nElements; assert rhog_consistent.shape[1] == nSpace
    
    for eN in range(nElements):
        JeT    = numpy.transpose(J[eN,0])    #assumed constant over element
        JeTinv = numpy.transpose(Jinv[eN,0])

        #gravity mapped back to reference element, [g_{\hat{x}},g_{\hat{y}}]^{T}
        ghat   = numpy.dot(JeT,gravity)

        #local degrees of freedom for concentration
        c_e = numpy.array([c_dof[c_l2g[eN,0]],c_dof[c_l2g[eN,1]],c_dof[c_l2g[eN,2]]])

        #compute \tilde{h}_{\hat{x}} = \sum_{k} h^k_{\hat{x}}\hat{N}_k, nodal projection of function
        # h_{\hat{x}} = \int_0^{\hat{x}}\rho(s,\hat{y})g_{\hat{x}}\ds
        # h^0_{\hat{x}}=h^2_{\hat{x}} = 0
        # h^1_{\hat{x}}= g_{\hat{x}}\int_0^{1}\rho(c(s,0))\ds

        if abs(c_e[1]-c_e[0]) <= eps_tol: #made this up, needs to be checked
            h1_xhat = ghat[0]*rho_eval(c_e[0])
        else:
            h1_xhat = ghat[0]*(rhoInt_eval(c_e[1])-rhoInt_eval(c_e[0]))/(c_e[1]-c_e[0])
        #could also just use simpson's rule and avoid rhoInt_eval?

        #compute \tilde{h}_{\hat{y}} = \sum_{k} h^k_{\hat{y}}\hat{N}_k, nodal projection of function
        # h_{\hat{y}} = \int_0^{\hat{y}}\rho(\hat{x},s)g_{\hat{y}}\ds
        # h^0_{\hat{y}}= h^1_{\hat{y}} = 0
        # h^2_{\hat{y}}= g_{\hat{y}}\int_0^{1}\rho(c(0,s))\ds

        if abs(c_e[2]-c_e[0]) <= eps_tol: #made this up, needs to be checked
            h2_yhat = ghat[1]*rho_eval(c_e[0])
        else:
            h2_yhat = ghat[1]*(rhoInt_eval(c_e[2])-rhoInt_eval(c_e[0]))/(c_e[2]-c_e[0])
        #could also just use simpson's rule and avoid rhoInt_eval?

        #local correction
        h_hat = numpy.array([h1_xhat,h2_yhat])
        
        rhog_consistent[eN] = numpy.dot(JeTinv,h_hat)

    #end element loop

#end function
        
# density function and its integral for consistent velocity approximation from Knabner-Frolkovic

def rho_eval(mf):
    return epsilon*mf
def rhoInt_eval(mf):
    return 0.5*epsilon*mf*mf


##################################################



aOfX = {0:hydraulicConductivity}

# fOfX = {0:fluidSourceTerms} Add for non-homogeneous case

rOfX = {0: reactionTerms}

bOfX = {0:bouyancySourceTerm}

consistentVelocityFunction=knabnerFrolkovicConsistentVelCorrection_Triangle_P1

analyticalSolution = flowAnalyticalSolution

analyticalSolutionVelocity = velocityAnalyticalSolution

if consistentVelocityFlag == 1:
    coefficients = GwvdCoefficients.VariableDensity_mod2_FlowEquationCoefficients( S=S, omega = omega,
                                                                                  epsilon=epsilon, nc = ncflow, nd = nd,
                                                                                  rho_fw=rho_fw, gc=gc, Kx=Kx, Ky =Ky, mu=nu, 
                                                                                  drho_dmf=drho_dmf,
                                                                                  consistentVelocityFlag = consistentVelocityFlag,
                                                                                  gravity = gravity,T_model=1)  
else:
    coefficients = GwvdCoefficients.VariableDensity_mod2_FlowEquationCoefficients(S=S, omega=omega,
                                                           epsilon=epsilon, nc=ncflow, nd= nd,rho_fw=rho_fw,
                                                           gc=gc, Kx=Kx, Ky =Ky, mu=nu, drho_dmf=drho_dmf, T_model=1)
  
initialConditions = {0:hydrostaticIC(mf_fw,mf_sw,L[1])}  #{0:constantIC(0.0)} #adjust back
dirichletConditions ={0:dummyBCs} # {0:flowHeadBCs}# 

fluxBoundaryConditions ={0: 'setFlow'} #{0:'setflow'} #{0:'noFlow'}#{0:'setFlow'} 

advectiveFluxBoundaryConditions = {0:getNoFlowBCs} #{0:gettopNoFlowBCs} # #{0:dummyBCs}  

diffusiveFluxBoundaryConditions ={0:{0:getNoFlowBCs}} #{0:{0:gettopNoFlowBCs}} # 


