from proteus.mprans import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem
from proteus.mprans import SpatialTools as st
import os


"""
This is a problem of a flow moving passed a cylindrical obstacle. To have portions
of the boundary be reflecting, the user can set the boundary key to 'reflecting'
and boundary key value to 99.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #

opts = Context.Options([
    ('sw_model', 0, "sw_model = {0,1} for {SWEs, Disperisve SWEs}}"),
    ("final_time", 10.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs", False, "Use reflecting BCs for all boundaries"),
    ("structured", False, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("mannings", 0.02, "Mannings roughness coefficient")
])

###################
# DOMAIN AND MESH #
###################
domain = Domain.PlanarStraightLineGraphDomain()

# define length and width of rectangle
length = 40.0
width = 10.0

# define vertices and segments
my_vertices = [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]]
my_segments = [[0, 1], [1, 2], [2, 3], [3, 0]]

# boundary tags dictionary
bt = {'inflow': 1, 'reflecting': 99}
my_vertexFlags = [bt['inflow'], bt['reflecting'], bt['reflecting'], bt['inflow']]
my_segmentFlags = [bt['reflecting'], bt['reflecting'], bt['reflecting'], bt['inflow']]

my_customShape = st.CustomShape(domain, boundaryTags=bt,
                           vertices=my_vertices, vertexFlags=my_vertexFlags,
                           segments=my_segments, segmentFlags=my_segmentFlags)

# define cylindrical obstacle. boundaryTag value is set to 99-2 here because of
# weird 'start_flag' issue in `_assembleGeometry` in SpatialTools.py
center_x = 10.0
center_y = 5.0
my_circle = st.Circle(domain=domain,
                      radius=1.,
                      barycenter=[center_x, center_y],
                      coords=[center_x, center_y],
                      nPoints=20,
                      boundaryTag={'reflecting': 99-2})

# set obstacle to be hole and set boundary tag to reflecting
my_circle.setHoles([[center_x, center_y]])

# assemble domain and set mesh triangle options
st.assembleDomain(domain)
domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
nnx = None
nny = None

# hack for using same mesh for tests, don't remove
if opts.he==4.0:
    from shutil import copyfile
    saved_mesh=os.path.join(os.path.dirname(os.path.abspath(__file__)), "for_testMeshes")
    for name in ["obstacle.1.neigh","obstacle.1.poly","obstacle.edge","obstacle.ele","obstacle.node","obstacle.poly"]:
        saved_file = os.path.join(saved_mesh,name)
        copied_to = os.path.join(os.getcwd(), name)     # the run's directory, not the checkout
        copyfile(saved_file, copied_to)
    domain.polyfile=os.path.join(os.getcwd(), "obstacle")
    domain.MeshOptions.genMesh=False
##########################################
# DEFINE INITIAL CONSTANTS AND FUNCTIONS #
##########################################
g = 9.81

# constants
h_initial = 0.75
h_inflow = 1.0
q_inflow = h_inflow * 1.5

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
L = (length, width)
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
                     'h_times_w': hw_at_t0(),
                     'h_times_beta':Zero()}
boundaryConditions = {'water_height': h_DBC,
                      'x_mom': x_mom_DBC,
                      'y_mom': y_mom_DBC,
                      'h_times_eta': heta_DBC,
                      'h_times_w': hw_DBC,
                      'h_times_beta':x_mom_DBC}
# if want to use reflecting conditions on all boundaries switch above to
# boundaryConditions = {'water_height': lambda x, flag: None,
#                       'x_mom': lambda x, flag: None,
#                       'y_mom': lambda x, flag: None,
#                       'h_times_eta': lambda x, flag: None,
#                       'h_times_w': lambda x, flag: None}

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
                                              bathymetry=bathymetry_function,
                                              genMesh=False)
mySWFlowProblem.physical_parameters['mannings'] = opts.mannings
