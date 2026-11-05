from proteus import *
from proteus.default_p import *

r"""
This is a 1D dam break (Riemann) problem on the flat domain [0,L] and time interval [0,T].

.. math ::

   h_t    + (h u)_x = 0  

  (h u)_t + (h u^2  + \frac{h^2}{2g} )_x  + ghb_x + \tau_b = 0

where the :math:`t` and :math:`x` subscripts denote derivatives in
time and space, :math:`h` is the water depth, and :math:`u` is the
velocity.  :math:`g` is the magnitude of gravity, :math:`b(x)` is the
bottom elevation (bathymetry)

:math:`\tau_b` is the bottom friction 

.. math:: 

    \tau_b = C_f u|u|
       C_f = g c_b/h^{c_a}

Here :math:`C_f` is the friction coefficient. The parameters :math:`c_b,c_a` are :math:`n^2` and 1/3, respectively if Manning's law is used.
 

The initial condition is


.. math ::
   h = h_R,  x <= L/2
       h_L,  x >  L/2
   u = 0

 
Wall boundary conditions are set on both sides

..math ::
  hu = 0, x \in \{0,L\}
  
Note this translates into a zero flux condition for the mass
conservation equation and zero Dirichlet condition for the momentum
equation.

The bottom friction is zero. Note that if the eddy viscosity is
non-zero, then the Numerical Flux and(possibly) the time integration
method choice need to be updated.

"""

#1d example
nd=1

#set the spatial domain
L=(10.0,1.0,1.0)
#set the tempora domain
T=20.0
#gravity magnitude
g = 9.8
#left and right values for the initial condition
HR=0.0
HL=1.0
#transport coefficients
coefficients = ShallowWater(g=g,
                            nd=nd)


#initial condition functions

#for depth
class DamBreakIC(object):
    def __init__(self,Lx,HL,HR):
        self.xc = Lx/2.0
        self.HL=HL
        self.HR=HR
    def uOfXT(self,x,t):
        import math
        if x[0]>self.xc:
            h = self.HL
        else:
            h = self.HR
        return h
#for momentum
class ZeroIC(object):
    def uOfXT(self,x,t):
        return 0.0

#set the initial conditions
initialConditions = {0:DamBreakIC(L[0],HL,HR),
                      1:ZeroIC()}

#Wall boundary conditions: no flux for mass conservation
def getDBC_h(x,flag):
    return None

#Zero Dirichlet conditions for momentum
def getDBC_hu(x,flag):
    if (x[0] < 1.0e-8 or
        x[0] > L[0] - 1.0e-8):
        return lambda x,t: 0.0
    else:
        return None
    
dirichletConditions = {0:getDBC_h,
                       1:getDBC_hu}

fluxBoundaryConditions = {0:'outFlow',
                          1:'outFlow'}

#Specify directly the advective flux boundary condition for the mass equation to be zero
def getAFBC_h(x,flag):
    if (x[0] < 1.0e-8 or
        x[0] > L[0] - 1.0e-8):
        return lambda x,t: 0.0
    else:
        return None
#no need to do anything for momentum    
def getAFBC_hu(x,flag):
    return None

advectiveFluxBoundaryConditions =  {0:getAFBC_h,
                                    1:getAFBC_hu}

diffusiveFluxBoundaryConditions = {0:{},1:{}}


