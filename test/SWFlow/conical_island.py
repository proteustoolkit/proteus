from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
from proteus import (Domain, Context,
                     MeshTools as mt)
from proteus.Profiling import logEvent
from proteus.Gauges import PointGauges
import numpy as np
import proteus.SWFlow.SWFlowProblem as SWFlowProblem


"""
We consider the 1995 laboratory experiments conducted by
[Briggs, 1995] at the US Army Waterways Experiment Station in
Vicksburg, Mississippi. The laboratory experiments were motivated by
several tsunami events in the 1990s where large unexpected run-up
heights were observed on the back (or lee) side of small
islands. There are two cases defined here, Case B and Case C.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 10.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("want_gauges", False, "Output for water height point gauge"),
    ("which_case", 0, "which_case = {0,1} for {case_C, case_B}")
])

###################
# DOMAIN AND MESH #
###################
L = (25.0, 30.0)  # this is length in x direction and y direction
refinement = opts.refinement
rectangle = RectangularDomain(L=L)

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = nnx
he = L[0]/float(nnx-1)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("island")
    domain = PlanarStraightLineGraphDomain(fileprefix="island")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

###############################
#  CONSTANTS NEEDED FOR SETUP #
###############################
g = 9.81

# stuff for solitary wave
h0 = 0.32
alpha = 0.181 * h0
xs = 7.56023
if opts.which_case == 1:
    alpha = 0.091 * h0
    xs = 6.81474

r = np.sqrt(3.*alpha/(4.*h0**2*(h0+alpha)))
c = np.sqrt(g * (h0 + alpha))

# stuff for cone bathymetry
htop = 0.625
rcone = 3.6
scone = 4.0
hcone = 0.9


#####################################
#   Some functions defined here    #
####################################
def solitary_wave(x, t):
    sechSqd = (1.0 / np.cosh(r * (x - xs)))**2
    return alpha * sechSqd


def bathymetry_function(X):
    x = X[0]
    y = X[1]
    radius = np.sqrt((x - 12.96)**2 + (y - 13.80)**2)
    conds = [radius <= rcone, radius > rcone]
    bath = [lambda radius: np.minimum(
        htop, hcone - radius / scone), lambda radius: 0.0]
    return np.piecewise(radius, conds, bath)

##############################
##### INITIAL CONDITIONS #####
##############################


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        hTilde = h0 + solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.)
        return h


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        hTilde = h0 + solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.)
        return h * c * (hTilde-h0)/hTilde


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
        hTilde = h0 + solitary_wave(X[0], 0)
        h = max(hTilde - bathymetry_function(X), 0.)
        hTildePrime = -2.0 * alpha * r * np.tanh(r * (X[0] - xs)) * sechSqd
        hw = -h**2 * c*h0*hTildePrime/hTilde**2
        return hw


class Zero(object):
    def uOfXT(self, X, t):
        return 0.


###############################
##### BOUNDARY CONDITIONS #####
###############################
X_coords = (0.0, L[0])  # this is x domain, used in BCs
Y_coords = (0.0, L[1])  # this is y domain, used in BCs


def x_mom_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda X, t: 0.0


def y_mom_DBC(X, flag):
    if X[1] == Y_coords[0] or X[1] == Y_coords[1]:
        return lambda X, t: 0.0


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
                      'h_times_beta': x_mom_DBC}
# **************************** #
# ********** GAUGES ********** #
# **************************** #
heightPointGauges = PointGauges(gauges=((('h',), ((xs, 16.05, 0),
                                                  (xs, 14.55, 0),
                                                  (xs, 13.05, 0),
                                                  (xs, 11.55, 0),
                                                  (9.36, 13.80, 0),
                                                  (10.36, 13.80, 0),
                                                  (12.96, 11.22, 0),
                                                  (15.56, 13.80, 0))),),
                                activeTime=(0.01, opts.final_time),
                                fileName='island_wave_gauges.csv')
# ********************************************* #
# ********** Create my SWFlowProblem ********** #
# ********************************************* #
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
                                              reflectingBCs=opts.reflecting_BCs,
                                              bathymetry=bathymetry_function,
                                              analyticalSolution=None)
mySWFlowProblem.physical_parameters['LINEAR_FRICTION'] = 0
mySWFlowProblem.physical_parameters['mannings'] = 0.0
if opts.want_gauges:
    mySWFlowProblem.auxiliaryVariables = [heightPointGauges]
