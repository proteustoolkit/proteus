#! /usr/bin/env python

r"""
common definitions for simple miscible displacement problem

    Flow (pressure equation)
    
    \deld \vec q = s(x,t)

          \vec q =-\frac{K(x,t)}{\hat{\mu}(c(x,t))} \grad h

    Transport equation      
    \pd{\theta c}{t} + \deld \left[\vec q c - \theta\ten{D}\nabla c\right] = s_c(x,t) 

    Here K is the Saturated Conductivity, K = k\varrho_0 |g|/\mu_0
      and
      \hat{\mu} = \mu/\mu_0
    where \mu is the dynamic viscosity, \varrho is the density, and the subscript 0 means a reference value
    
     We'll assume \hat{\mu}(c) = a*(c-c_0) + b
     
"""
from proteus import *
from proteus import Context
from proteus import SubsurfaceTransportCoefficients as STC

import numpy as np

# Context options (parun ... -C "unstructured_mesh=False refinement_level=8 ndays=500 nDTout=2")
ct = Context.Options([
    ("unstructured_mesh", True, "triangle mesh of the poly file; False uses a regular triangular mesh (no mesh generator)"),
    ("refinement_level", 32, "number of mesh intervals along the 1000 m side"),
    ("ndays", 5000.0, "final time [d]"),
    ("nDTout", 50, "number of output steps"),
])

class MiscibleDisplacementCoefficients_Flow(STC.SinglePhaseDarcyCoefficients):
    r"""
    Simplified transport coefficients for pressure equation in single phase flow with mild viscosity depedence

    Flow (pressure equation)
    
    \deld \vec q = s(x,t)

          \vec q =-\frac{K(x,t)}{\hat{\mu}(c(x,t))} \grad h

    Base class does most of the work for evaluating coefficients
    """

    def __init__(self,
                 K_types,     #conductivity function (x,t) that varies by element type
                 source_types,#source function (x,t) that varies by element type                 
                 nd=2,        #number of space dimensions
                 viscosity_a=0.0, viscosity_b=1.0,#mu = a*c + b
                 visc_ref_conc= 0.0,
                 concentration_model_id=None,  #which number is the model for the concentration transport equation? 
                 timeVaryingCoefficients=False, #do the coefficients vary in time?
                 materialValuesLocallyConstant=False): #are the material functions constants? e.g., K_j(x,t) = K^0_j ?
        if concentration_model_id is not None and viscosity_a <= 1.0e-16:
            print("Warning, specified concentration model with id {0} but no viscosity dependence mu=a*c+b with a={1:10.3e} b={2:10.3e} ".format(concentration_model_id,viscosity_a,viscosity_b))
        if concentration_model_id is None and viscosity_a > 1.0e-16:
            print("Warning, no specified concentration model but have viscosity dependence mu=a*c+b with a={0:10.3e} b={1:10.3e} ".format(viscosity_a,viscosity_b))
        
        self.concentration_model_id = concentration_model_id; self.concentration_model = None
        self.viscosity_a = viscosity_a 
        self.viscosity_b = viscosity_b 
        self.visc_ref_conc= visc_ref_conc
        STC.SinglePhaseDarcyCoefficients.__init__(self,K_types,source_types,nc=1,nd=nd,
                                                  timeVaryingCoefficients=timeVaryingCoefficients,
                                                  materialValuesLocallyConstant=materialValuesLocallyConstant)
        
        self.variableNames=['h']
        
    def attachModels(self,modelList):
        if self.concentration_model_id is not None: #grab a reference to the model that solves for concentration 
            assert 0 <= self.concentration_model_id and self.concentration_model_id < len(modelList)
            self.concentration_model = modelList[self.concentration_model_id]
            #assumes that the first unknown in the transport equation is the concentration of our species of interest
            #get references to the quadrature dictionaries ...
            #element quadrature points
            self.q_c    = self.concentration_model.q[('u',0)]
            #exterior boundary of domain
            self.ebqe_c = self.concentration_model.ebqe[('u',0)]
            #element boundary points treated as unique per element (e.g., for DG) 
            if ('u',0) in self.concentration_model.ebq:
                self.ebq_c = self.concentration_model.ebq[('u',0)]
            #element boundary points treated as unique per element boundary
            if ('u',0) in self.concentration_model.ebq_global:
                self.ebq_global_c = self.concentration_model.ebq_global[('u',0)]

    def initializeElementQuadrature(self,t,cq):
        #call parent classes function, then initialize quadrature point for concentration and viscosity values
        STC.SinglePhaseDarcyCoefficients.initializeElementQuadrature(self,t,cq)
        self.q_c = np.zeros(cq[('u',0)].shape,'d')
        self.q_mu = np.zeros(cq[('u',0)].shape,'d')
        self.q_mu.fill(self.viscosity_b)
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        #call parent classes function, then initialize element boundary quadrature point for concentration and viscosity  values
        STC.SinglePhaseDarcyCoefficients.initializeElementBoundaryQuadrature(self,t,cebq,cebq_global)
        #spatial coordinates have an 'extra' dimension (3). Don't always have a 'u' value
        ebq_shape = cebq['x'].shape[:-1];         ebq_global_shape = cebq_global['x'].shape[:-1]
        self.ebq_c = np.zeros(ebq_shape,'d');  self.ebq_global_c = np.zeros(ebq_global_shape,'d')
        self.ebq_mu= np.zeros(ebq_shape,'d');  self.ebq_global_mu= np.zeros(ebq_global_shape,'d')
        self.ebq_mu.fill(self.viscosity_b);       self.ebq_global_mu.fill(self.viscosity_b)

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        #call parent classes function, then initialize quadrature point for concentration and viscosity values
        STC.SinglePhaseDarcyCoefficients.initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe)
        self.ebqe_c = np.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_mu = np.zeros(cebqe[('u',0)].shape,'d')
        self.ebqe_mu.fill(self.viscosity_b)

    def evaluate(self,t,c):
        #takes care of K(x,t) and f(x,t)
        STC.SinglePhaseDarcyCoefficients.evaluate(self,t,c)
        #figure out which set of quadrature points this is
        if c['x'].shape == self.q_x_shape:
            mu = self.q_mu; conc = self.q_c;
        elif c['x'].shape == self.ebqe_x_shape:
            mu = self.ebqe_mu; conc = self.ebqe_c;
        elif c['x'].shape == self.ebq_global_x_shape:
            mu = self.ebq_global_mu; conc = self.ebq_global_c;
        elif c['x'].shape == self.ebq_x_shape:
            mu = self.ebq_mu; conc = self.ebq_c;
        else:
            raise NotImplementedError
        #mu = a*(c-c_0) + b
        mu[:] = conc; mu -= self.visc_ref_conc; mu[mu < 0] = 0.0  #trim negative values
        mu *= self.viscosity_a; mu += self.viscosity_b
        #mwf debug
        #print "Flow evaluate t={} mu_a= {} mu_b= {} conc.max() = {} conc.min= {} mu.max()= {} mu.min()= {} ".format(t,self.viscosity_a,self.viscosity_b,conc.max(),conc.min(),
        #                                                                                                                       mu.max(),mu.min())

        #check for divide by zero
        assert np.absolute(mu).min() > 0.0, "problem, viscosity has a zero value about to divide by min |mu| = {}".format(np.absolute(mu).min())
        #
        for i in range(c[('a',0,0)].shape[-1]):
            np.divide(c[('a',0,0)][...,i],mu,c[('a',0,0)][...,i])

    #eval
#Miscible Displacement Class


name = "MiscibleDisplacement"

### Spatial Domain ###
L = (1000,1000,1) #m
nd= 2


#temporal domain
#for converting from seconds to days
days2sec = 60.*60.*24.

ndays = ct.ndays
T = ndays #[d]
nDTout = ct.nDTout

#just a simple rectangular domain for now
regular_domain = Domain.RectangularDomain(L[:nd],
                                          units="m")
unstructured_mesh = ct.unstructured_mesh
if unstructured_mesh:
    #generate a .poly file for triangle
    regular_domain.writePoly('mesh')
    #'domain' is the name of the object that Proteus looks for to define geometry
    domain = Domain.PlanarStraightLineGraphDomain(fileprefix='mesh',name='md_mesh')
    domain.boundaryTags = regular_domain.boundaryLegend
else:
    #regular triangular mesh of the rectangle, nnx x nny nodes
    domain = regular_domain


genMesh = True
refinement_level = ct.refinement_level #define characteristic length
he = L[0]/float(refinement_level)

triangleOptions = "VApq30Dena{area:8.8f}".format(area=(he**2)/2.0)


### Material Properties ###
#homogeneous for now
nmaterial = 1
K_0 = 4.0# m/d
f_0 = 0.0#1/d 
porosity_0 = 0.35 #[-]
alpha_L_0   = 0.01 #[m]
alpha_T_0   = 0.001 #[m]

Ident = np.zeros((nd,nd),'d')
Ident[0,0]=1.0; Ident[1,1] = 1.0

conductivities = {1:lambda x,t: K_0*Ident}
sources = {1:lambda x,t: f_0}
if not unstructured_mesh:
    #the regular mesh's elements have material type 0
    conductivities[0] = conductivities[1]
    sources[0] = sources[1]
#transport heterogeneity is handled as constants only
omega_types   = np.array([porosity_0, porosity_0])
alpha_L_types = np.array([alpha_L_0,alpha_L_0])
alpha_T_types = np.array([alpha_T_0,alpha_T_0])

#check about this
#sources[0] = sources[1]
#conductivities[0] = conductivities[1]
#constant viscosity
mu_b= 1.0 #[-] 
mu_a= 5.0

d_mol = np.array([1.3e-9])

### Boundary Conditions ###
## Flow Equation
#Pressure boundary condition on bottom left and top right
#no flow everywhere else
inflow_length = L[1]*0.1
outflow_length= L[1]*0.8

#piezometric head
head_inflow = 100.0 #[m]
head_outflow= 0.0

## Transport Equation
concentration_inflow = 1.0 # kg/m^3 ?
concentration_background = 0.0 
#
def head_bc(x,flag):
    if flag == domain.boundaryTags['left'] and x[1] <= inflow_length:
        return lambda x,t: head_inflow
    if flag == domain.boundaryTags['right'] and x[1] >= outflow_length:
        return lambda x,t: head_outflow

def noflux(x,flag):
    if flag in [domain.boundaryTags['bottom'],domain.boundaryTags['top']]:
        return lambda x,t: 0.0
    if flag == domain.boundaryTags['left'] and x[1] > inflow_length:
        return lambda x,t: 0.0
    if flag == domain.boundaryTags['right'] and x[1] < outflow_length:
        return lambda x,t: 0.0


def advective_flux(x,flag):
    if flag in [domain.boundaryTags['bottom'],domain.boundaryTags['top']]:
        return lambda x,t: 0.0
    if flag == domain.boundaryTags['left'] and x[1] > inflow_length:
        return lambda x,t: 0.0
    if flag == domain.boundaryTags['right'] and x[1] < outflow_length:
        return lambda x,t: 0.0
    if flag == domain.boundaryTags['right'] and x[1] < outflow_length:
        return None

def dispersive_flux(x,flag):
    if flag in [domain.boundaryTags['bottom'],domain.boundaryTags['top'],
                domain.boundaryTags['left'], domain.boundaryTags['right']]:
        return lambda x,t: 0.0

#
def concentration_bc(x,flag):
    if flag == domain.boundaryTags['left'] and x[1] <= inflow_length:
        return lambda x,t: concentration_inflow

### Initial conditions ###
class ConstantIC(object):
    def __init__(self,val=0.0):
        self.val = val
    def uOfXT(self,x,t):
        return self.val

##Flow Equation
initialConditions_flow = {0:ConstantIC(head_outflow)}
##Transport Equation
initialConditions_trans = {0:ConstantIC(concentration_background)}


# numerics
parallel = False

nnx = nny = int(L[0]/he) + 1
nLevels = 1
if parallel:
    nLevels = 1

# parallel partitioning
if parallel:
    nLayersOfOverlapForParallel = 1     
    parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
    numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior

# 
gw_quad_order = 3
