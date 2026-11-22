from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem
from proteus import SpatialTools as st


"""
This is a mach reflection test.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #

opts = Context.Options([
    ('sw_model', 0, "sw_model = {0,1} for {SWEs, Disperisve SWEs}}"),
    ("final_time", 5.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs", False, "Use reflecting BCs"),
    ("structured", False, "Structured or unstructured mesh"),
    ("he", 0.3, "Mesh size for unstructured mesh"),
    ("mannings", 0.0, "Mannings roughness coefficient")
])

###################
# DOMAIN AND MESH #
###################
domain = Domain.PlanarStraightLineGraphDomain()

my_vertices = [[0.0,0.0],[3.0,0.0],[10.0,3.2],[10.0,10.0],[0.0,10.0]]
my_segments = [[0,1],[1,2],[2,3],[3,4],[4,0]]

# boundary tags dictionary
bt = {'inflow': 1, 'reflecting': 99}
my_vertexFlags = [bt['inflow'], bt['reflecting'], bt['reflecting'], bt['reflecting'],
                  bt['reflecting']]
my_segmentFlags = [bt['reflecting'], bt['reflecting'], bt['reflecting'], bt['reflecting'],
                  bt['inflow']]
my_customShape = st.CustomShape(domain, boundaryTags=bt,
                           vertices=my_vertices, vertexFlags=my_vertexFlags,
                           segments=my_segments, segmentFlags=my_segmentFlags)

st.assembleDomain(domain)
domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
nnx = None
nny = None

##########################################
# DEFINE INITIAL CONSTANTS AND FUNCTIONS #
##########################################
g = 9.81

# constants
h_initial = 0.1
h_inflow = 3.0 / 2.0 * h_initial
q_inflow = h_inflow * 2.0 / 3.0

def bathymetry_function(X):
    x = X[0]
    return 0.0 * x


##############################
#    INITIAL CONDITIONS      #
##############################


class water_height_at_t0(object):
    def uOfXT(self, X, t):
        x = X[0]
        h = h_initial - bathymetry_function(X)
        return max(h, 0.)


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        return 0.0


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

class Zero(object):
    def uOfXT(self, x, t):
        return 0.0

###############################
#     BOUNDARY CONDITIONS     #
###############################
L = (10.0, 10.0)
X_coords = (0.0, L[0])
Y_coords = (0.0, L[1])

def h_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda x, t: h_inflow

def x_mom_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda x, t: q_inflow

def y_mom_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda x, t: 0.0

def heta_DBC(X, flag):
    if X[0] == X_coords[0]:
        return lambda x, t: h_inflow**2

def hw_DBC(X, flag):
    if X[0] == X_coords[0]:
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
                     'h_times_w': hw_at_t0()}
boundaryConditions = {'water_height': h_DBC,
                      'x_mom': x_mom_DBC,
                      'y_mom': y_mom_DBC,
                      'h_times_eta': heta_DBC,
                      'h_times_w': hw_DBC}

mySWFlowProblem = SWFlowProblem.SWFlowProblem(sw_model=opts.sw_model,
                                              cfl=opts.cfl,
                                              outputStepping=outputStepping,
                                              structured=opts.structured,
                                              he=opts.he,
                                              nnx=nnx,
                                              nny=nny,
                                              domain=domain,
                                              initialConditions=initialConditions,
                                              boundaryConditions=boundaryConditions,
                                              reflectingBCs=opts.reflecting_BCs,
                                              bathymetry=bathymetry_function)
mySWFlowProblem.physical_parameters['mannings'] = opts.mannings