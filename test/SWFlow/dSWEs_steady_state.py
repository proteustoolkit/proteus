from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain
import numpy as np
from proteus import (Domain, Context,
                     MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

"""
This is the steady state problem for to the dispersive Serre-Green-Nagdhi
equations (or dispersive shallow water equations).
We use the Bernoulli relation to derive the bathymetry profile where
we assume the water height is given by h = h0 + a h0 sech(r*(x-x0))^2.
The variables are defined below. See 'Hyperbolic Relaxation Technique
For Solving The Dispersive Serre Equations With Topography'
by Guermond, Kees, Popov, Tovar for more details. Note that this is a fake
1D problem, ie we are doing simulation in 2d but only consider x dominated flow.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 40, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level")
])

###################
# DOMAIN AND MESH #
###################
L = (20.0, 1.0)
refinement = opts.refinement
domain = RectangularDomain(L=L, x=[-10.0, 0, 0])

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = (nnx - 1)//10 + 1
he = L[0]/float(nnx-1)
triangleOptions = "pAq30Dena%f" % (0.5 * he**2,)

###############################
#  CONSTANTS NEEDED FOR SETUP #
###############################
g = 9.81
h0 = 1.0  # this is still water depth
a = 0.2  # this is amplitude
q = np.sqrt(g * h0**3 * (1.0 + a) / 2)  # this is flow rate
r = 1.0/h0 * np.sqrt(3.0 * a / (1.0 + a))  # this is solitary wave width
x0 = 0  # wave is centered at x = 0

###############################
#   Functions defined here    #
###############################
def solitary_wave(x, t):
    sechSqd = 1.0/np.cosh(r*(x-x0))**2
    return h0 + a * h0 * sechSqd


def bathymetry_function(X):
    x = X[0]
    sechSqd = 1.0/np.cosh(r*(x-x0))**2
    waterh = h0 * (1.0 + a * sechSqd)
    z = -1.0/2.0 * (waterh - h0)
    return z

##############################
##### INITIAL CONDITIONS #####
##############################
class water_height_at_t0(object):
    def uOfXT(self, X, t):
        h = solitary_wave(X[0], 0)
        return h

class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        return q

class x_mom_exact(object):
    def uOfXT(self, X, t):
        return q

class y_mom_at_t0(object):
    def uOfXT(self, X, t):
        return 0.

"""
heta and hw are needed for the hyperbolic serre-green-naghdi equations.
For initial conditions, heta -> h^2, hbeta->q(dot)grad(Z), hw -> h^2div(u)+3/2*hbeta.
It's often okay to take hbeta=0. Note that the BCs for the heta and hw should be same as h
and BCs for hbeta should be same as x_mom.
For more details see: 'Hyperbolic relaxation technique for solving the dispersive Serre Equations
with topography' by Guermond, Popov, Tovar, Kees.
"""
class heta_at_t0(object):
    def uOfXT(self, X, t):
        h = water_height_at_t0().uOfXT(X, t)
        return h**2


class hw_at_t0(object):
    def uOfXT(self, X, t):
        x = X[0]
        sechSqd = 1.0/np.cosh(r*(x-x0))**2
        hw = -a * h0 * q * r * (2.0 + 3.0*(-0.50)) * sechSqd * np.tanh(r * x)
        return hw


class hbeta_at_t0(object):
    def uOfXT(self, X, t):
        x = X[0]
        sechSqd = 1.0/np.cosh(r*(x-x0))**2
        hbeta = -2.0 * a * h0 * r * q * (-1.0/2.0) * sechSqd * np.tanh(r * x)
        return hbeta


###############################
##### BOUNDARY CONDITIONS #####
###############################
X_coords = (-10.0, 10.0)  # this is domain, used in BCs


def water_height_DBC(X, flag):
    if (opts.sw_model==1):
        if X[0] == X_coords[0]:
            return lambda x, t: water_height_at_t0().uOfXT(X, 0.0)
        elif X[0]==X_coords[1]:
            return lambda x,t: water_height_at_t0().uOfXT(X ,0.0)
    if (opts.sw_model==0):
        if X[0] == X_coords[1]:
            return lambda x, t: water_height_at_t0().uOfXT(X, 0.0)

def x_mom_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda X, t: x_mom_exact().uOfXT(X, 0.0)

def heta_DBC(X, flag):
    if X[0] == X_coords[0] or X[0]==X_coords[1]:
        return lambda x,t: heta_at_t0().uOfXT(X, 0.0)

def hw_DBC(X, flag):
    if X[0] == X_coords[0] or X[0]==X_coords[1]:
        return lambda x, t: hw_at_t0().uOfXT(X, 0.0)

def hbeta_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda X, t: hbeta_at_t0().uOfXT(X, 0.0)

# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #
outputStepping = SWFlowProblem.OutputStepping(opts.final_time, dt_output=opts.dt_output)
initialConditions = {'water_height': water_height_at_t0(),
                     'x_mom': x_mom_at_t0(),
                     'y_mom': y_mom_at_t0(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0(),
                     'h_times_beta': hbeta_at_t0()}
boundaryConditions = {'water_height': water_height_DBC,
                      'x_mom': x_mom_DBC,
                      'y_mom': lambda x, flag: lambda x, t: 0.0,
                      'h_times_eta': heta_DBC,
                      'h_times_w': hw_DBC,
                      'h_times_beta': hbeta_DBC}
mySWFlowProblem = SWFlowProblem.SWFlowProblem(sw_model=opts.sw_model,
                                              cfl=opts.cfl,
                                              outputStepping=outputStepping,
                                              structured=True,
                                              he=he,
                                              nnx=nnx,
                                              nny=nny,
                                              domain=domain,
                                              initialConditions=initialConditions,
                                              boundaryConditions=boundaryConditions,
                                              bathymetry=bathymetry_function)
mySWFlowProblem.physical_parameters['mannings'] = 0.0
