from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

"""
This is the problem of a solitary wave run-up on a sloping beach.
Experiments were reported in 'The runup of solitary waves' [Synolakas, 1987].
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #

# Note that the final time should correspond to the experimental data time
# Tstar = {10,15,20,...,65} for the final time. We redefine the final time
# in the problem set up at the bottom of the file
opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs, Disperisve SWEs}"),
    ("final_time", 30.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("mannings", 0.016, "Mannings roughness coefficient")
])

###################
# DOMAIN AND MESH #
###################
L = (50.0, 1.0)
refinement = opts.refinement
rectangle = RectangularDomain(L=L, x=[-35.0, 0, 0]) # x is origin

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = (nnx - 1)//20 + 1
he = L[0]/float(nnx-1)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("runup")
    domain = PlanarStraightLineGraphDomain(fileprefix="runup")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

###############################
#  CONSTANTS NEEDED FOR SETUP #
###############################
g = 9.81  # gravity
h0 = 1.0  # water depth
a = 0.28  # relative amplitude
slope = 1.0 / 19.85  # beach slope
k_wavenumber = np.sqrt(3.0 * a / (4.0 * h0**3))  # wavenumber
z = np.sqrt(3.0 * a * h0) / (2.0 * h0 * np.sqrt(h0 * (1.0 + a)))  # width of solitary wave
L_wave = 2.0 / k_wavenumber * np.arccosh(np.sqrt(20.0))  # wavelength of solitary wave
c = np.sqrt(g * (1.0 + a) * h0)  # wave celerity (or speed)
x0 = - h0 / slope - L_wave / 2.0  # location of the toe of the beach

###############################
#   Functions defined here    #
###############################
def solitary_wave(x, t):
    sechSqd = (1.00 / np.cosh(z * (x - x0 - c * t)))**2.00
    return a * h0 * sechSqd


def bathymetry_function(X):
    x = X[0]
    return np.maximum(slope * x, -h0)


class Zero(object):
    def uOfXT(self, x, t):
        return 0.0


######################
# INITIAL CONDITIONS #
######################


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        hTilde = solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.0)
        return h


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        hTilde = solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.0)
        return h * c * hTilde / (h0 + hTilde)


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
        hTilde = solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.0)
        hTildePrime = -2.0 * z * hTilde * np.tanh(z * (X[0] - x0 - c * t))
        hw = -h**2 * (c * h0 * hTildePrime / (h0 + hTilde)**2)
        return hw

###############################
##### BOUNDARY CONDITIONS #####
###############################
X_coords = (-35.0, 15.0)
Y_coords = (0.0, 1.0)

def x_mom_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda x, t: 0.0
def y_mom_DBC(X, flag):
    if X[1] == Y_coords[0] or X[1] == Y_coords[1]:
        return lambda x, t: 0.0


def water_height_DBC(X, flag):
    return None

def x_mom_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda X, t: 0.0

# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #
# redefine time here
T = opts.final_time
Tstar = T * np.sqrt(h0 / g)
outputStepping = SWFlowProblem.OutputStepping(
    Tstar, dt_output=opts.dt_output)
initialConditions = {'water_height': water_height_at_t0(),
                     'x_mom': x_mom_at_t0(),
                     'y_mom': y_mom_at_t0(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0()}
boundaryConditions = {'water_height': lambda x, flag: None,
                      'x_mom': x_mom_DBC,
                      'y_mom': y_mom_DBC,
                      'h_times_eta': lambda x, flag: None,
                      'h_times_w': lambda x, flag: None}

mySWFlowProblem = SWFlowProblem.SWFlowProblem(sw_model=opts.sw_model,
                                              cfl=opts.cfl,
                                              outputStepping=outputStepping,
                                              structured=opts.structured,
                                              he=he,
                                              nnx=nnx,
                                              nny=nny,
                                              domain=domain,
                                              initialConditions=initialConditions,
                                              boundaryConditions=boundaryConditions,
                                              bathymetry=bathymetry_function)
mySWFlowProblem.physical_parameters['mannings'] = opts.mannings
