from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
from proteus.Gauges import PointGauges
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

"""
Well-balancing test with conical island bathymetry and H_0 = 1m.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 10.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.33, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.1, "Mesh size for unstructured mesh")
])

###################
# DOMAIN AND MESH #
###################
L = (30.0, 25.0)  # this is length in x direction and y direction
refinement = opts.refinement
rectangle = RectangularDomain(L=L)

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = (nnx - 1)//2 + 1
he = L[0]/float(nnx-1)
triangleOptions = "pAq30Dena%f" % (0.5 * he**2,)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("well_balancing")
    domain = PlanarStraightLineGraphDomain(fileprefix="well_balancing")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

###############################
#  CONSTANTS NEEDED FOR SETUP #
###############################
g = 9.81
h0 = 1.0

# stuff for cone bathymetry
htop = 0.625
rcone = 3.6
scone = 4.0
hcone = 0.9


#####################################
#   Some functions defined here    #
####################################

def bathymetry_function(X):
    x = X[0]
    y = X[1]
    radius = np.sqrt((x - 12.96)**2 + (y - 13.80)**2)

    conds = [radius <= rcone, radius > rcone]
    bath = [lambda radius: np.minimum(
        htop, hcone-radius/scone), lambda radius: 0.0]
    return 0. * x  # np.piecewise(radius, conds, bath)

##############################
##### INITIAL CONDITIONS #####
##############################


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        return max(h0 - bathymetry_function(X), 0.)


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        return 0.


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
        return 0.


class hbeta_at_t0(object):
    def uOfXT(self, X, t):
        return 0.


###############################
##### BOUNDARY CONDITIONS #####
###############################
X_coords = (0.0, 30.0)  # this is x domain, used in BCs
Y_coords = (0.0, 25.0)  # this is y domain, used in BCs


def h_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda x, t: h0


def x_mom_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda x, t: 0.0


def y_mom_DBC(X, flag):
    if X[1] == Y_coords[0] or X[1] == Y_coords[1]:
        return lambda x, t: 0.0


def heta_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda x, t: h0**2


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
                     'h_times_beta': hw_at_t0()}
boundaryConditions = {'water_height': h_DBC,
                      'x_mom': x_mom_DBC,
                      'y_mom': y_mom_DBC,
                      'h_times_eta': heta_DBC,
                      'h_times_w': x_mom_DBC,
                      'h_times_beta': x_mom_DBC}
analytical_Solution = {'h_exact': water_height_at_t0(),
                       'hu_exact': x_mom_at_t0(),
                       'hv_exact': y_mom_at_t0(),
                       'heta_exact': heta_at_t0(),
                       'hw_exact': hw_at_t0(),
                       'hbeta_exact': hbeta_at_t0()}
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
                                              analyticalSolution=analytical_Solution)
mySWFlowProblem.physical_parameters['mannings'] = 0.0
