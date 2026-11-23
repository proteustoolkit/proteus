from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem


"""
This is a simple benchmark of a solitary wave propagating over a flat bottom.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #

opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs, Disperisve SWEs}}"),
    ("final_time", 5.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.1, "Mesh size for unstructured mesh")
])

###################
# DOMAIN AND MESH #
###################
L = (25.0, 2.0)
origin = [0, 0, 0]
refinement = opts.refinement
rectangle = RectangularDomain(L=L, x=origin)

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = (nnx - 1)//10 + 1
he = L[0]/float(nnx-1)
triangleOptions = "pAq30Dena%f" % (0.5 * he**2,)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("solitary")
    domain = PlanarStraightLineGraphDomain(fileprefix="solitary")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

##################################
# SOLITARY WAVE FUCTION AND BATH #
##################################
g = 9.81

# constants for solitary wave
h0 = 0.5
alpha = 0.2 * h0
xs = 5.0
r = np.sqrt(3.*alpha/(4.*h0**2*(h0+alpha)))
c = np.sqrt(g * (h0 + alpha))


def solitary_wave(x, t):
    sechSqd = (1.0 / np.cosh(r * (x - xs - c * t)))**2
    return alpha * sechSqd


def bathymetry_function(X):
    x = X[0]
    # return vector of zeros
    return x * 0.0


##############################
#    INITIAL CONDITIONS      #
##############################


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        h = h0 + solitary_wave(X[0], 0)
        return h


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        h = h0 + solitary_wave(X[0], 0)
        return h * c * (h-h0)/h


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
        sechSqd = (1.0 / np.cosh(r * (X[0] - xs)))**2.0
        h = h0 + solitary_wave(X[0], 0)
        hPrime = -2.0 * alpha * r * np.tanh(r * (X[0] - xs)) * sechSqd
        hw = -h**2 * c*h0*hPrime/h**2
        return hw

###################################
#    FOR ANALYTICAL SOLUTIONS     #
###################################


class Zero(object):
    def uOfXT(self, x, t):
        return 0.0


class water_height_at_tfinal(object):
    def uOfXT(self, X, t):
        return h0 + solitary_wave(X[0], opts.final_time)


###############################
#     BOUNDARY CONDITIONS     #
###############################
X_endPoints = (origin[0], L[0])
Y_endPoints = (origin[1], L[1])


def x_mom_DBC(X, flag):
    if X[0] == X_endPoints[0] or X[0] == X_endPoints[1]:
        return lambda x, t: 0.0


def y_mom_DBC(X, flag):
    if X[1] == Y_endPoints[0] or X[1] == Y_endPoints[1]:
        return lambda x, t: 0.0


def hbeta_DBC(X, flag):
    return lambda x, t: 0.0

# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #


outputStepping = SWFlowProblem.OutputStepping(
    opts.final_time, dt_output=opts.dt_output)
initialConditions = {'water_height': water_height_at_t0(),
                     'x_mom': x_mom_at_t0(),
                     'y_mom': y_mom_at_t0(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0(),
                     'h_times_beta': Zero()}
boundaryConditions = {'water_height': lambda x, flag: None,
                      'x_mom': x_mom_DBC,
                      'y_mom': y_mom_DBC,
                      'h_times_eta': lambda x, flag: None,
                      'h_times_w': lambda x, flag: None,
                      'h_times_beta': hbeta_DBC}
analytical_Solution = {'h_exact': water_height_at_tfinal(),
                       'hu_exact': Zero(),
                       'hv_exact': Zero(),
                       'heta_exact': Zero(),
                       'hw_exact': Zero(),
                       'hbeta_exact': Zero()}

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
                                              bathymetry=bathymetry_function,
                                              analyticalSolution=analytical_Solution)
