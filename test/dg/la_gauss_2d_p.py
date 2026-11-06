from proteus import *
from proteus.default_p import *
from math import *
"""
2D, Linear advection of a guassian
"""

##  \page Tests Test Problems 
# \ref la_gauss_2d_p.py "Linear advection of a Gaussian"
# \addtogroup test
#
#  \file la_gauss_2d_p.py
# @{
#  

##\ingroup test
#  \brief Conservative linear advection of a cone in a rotating
#  velocity field.
#
# \f{eqnarray*}
# \phi_t + \nabla \cdot (\vec u \phi) &=& 0 \\ 
# \Omega &=& [0,1] \times [0,1] \\
#  u^{x} &=&  2\pi(y-1/2)\\
#  u^{y} &=&  2\pi(1/2-x)\\
# \phi^{ex}(x,y,t) &=& \exp\left(-\frac{\|\vec x - \vec x_c\|^2}{2\sigma^2}\right)
#  \f}
#
# where 
# \f$\bar{x} = x - x_c\f$, \f$\bar{y} = y - y_c\f$, and
# \f$ x_c = \sin(2\pi t)/4 + 1/2\f$, \f$\; y_c = \cos(2\pi t)/4 + 1/2 \f$
#
# \image html  save_la_gauss_2d_exact.jpg "exact solution, T=0.75" 
# \image latex save_la_gauss_2d_exact.eps "exact solution" 
# \image html  save_la_gauss_2d_dgp2_soln.jpg "RKDG P^2 solution, Cr=0.15, L^2 error= 8.3e-4" 
# \image latex save_la_gauss_2d_dgp2_soln.eps "RKDG $P^2$ solution, Cr=0.15, $L^2$ error= 8.3e-4" 
#

nd = 2

name = 'la_gauss_2d_np1'

class RotatingGaussian2D:
    def __init__(self,sigma=1./8.):
        self.sigma = sigma
        self.xc= 0.65
        self.yc= 0.65
    def uOfXT(self,x,t):
        centerX = 0.25*sin(2*pi*t) + 0.5
        centerY = 0.25*cos(2*pi*t) + 0.5
        d2 = (x[0]-centerX)**2 + (x[1]-centerY)**2
        return exp(-0.5*d2/self.sigma**2)

class ConstantVelocityGaussian2D:
    def __init__(self,sigma=1./8.,b=[1.0,0.0]):
        self.sigma = sigma
        self.xc= 0.25
        self.yc= 0.5
        self.b = b
    def uOfXT(self,x,t):
        centerX = self.xc + self.b[0]*t
        centerY = self.yc + self.b[1]*t
        d2 = (x[0]-centerX)**2 + (x[1]-centerY)**2
        return exp(-0.5*d2/self.sigma**2)

#rotating
analyticalSolution = {0:RotatingGaussian2D(1.0/16.0)}

class UnitSquareRotation(TransportCoefficients.TC_base):
    from proteus.ctransportCoefficients import unitSquareRotationEvaluate
    # staticmethod: Cython 3 functions bind self like Python functions (C47)
    unitSquareRotationEvaluate = staticmethod(unitSquareRotationEvaluate)
    def __init__(self):
        mass={0:{0:'linear'}}
        advection={0:{0:'linear'}}
        diffusion={}
        potential={}
        reaction={}
        hamiltonian={}
        TransportCoefficients.TC_base.__init__(self,
                                             1,
                                             mass,
                                             advection,
                                             diffusion,
                                             potential,
                                             reaction,
                                             hamiltonian)
    def evaluate(self,t,c):
        self.unitSquareRotationEvaluate(c['x'],
                                        c[('u',0)],
                                        c[('m',0)],c[('dm',0,0)],
                                        c[('f',0)],c[('df',0,0)])
coefficients = UnitSquareRotation()

coefficients.variableNames=['u']

#now define the Dirichlet boundary conditions

def getDBC(x,tag):
    if (x[0] <= 0.0 or x[0] >= L[0] or
        x[1] <= 0.0 or x[1] >= L[1]):
        return lambda x,t: 0.0

dirichletConditions = {0:getDBC}

initialConditions  = {0:analyticalSolution[0]}


advectiveFluxBoundaryConditions =  {}

diffusiveFluxBoundaryConditions = {0:{}}

#rotation
T = 1.0

## @}