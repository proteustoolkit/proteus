from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

"""
This is a circular dam break problem used to compare the physics of the two  swflow models.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    # Simulation options
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 20., "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    # Mesh options
    ("refinement", 4, "Refinement level"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("reflecting_BCs", True, "Use reflecting BCs"),
    # Problem specific options
    ("mannings", 0.0, "Mannings roughness coefficient"), # usually = 0.
    ("still_water_depth", 1., "Depth of still water above floor"),
    ("dam_height", 0.8, "Height of water dam above still water"),
    ("dam_center", (10., 10.), "Center position of circular dam"),
    ("dam_radius", 2., "Radius of circular dam"),
])

###################
# DOMAIN AND MESH #
###################
L = (20.0, 20.0)  # this is length in x direction and y direction
refinement = opts.refinement
rectangle = RectangularDomain(L=L)

# CREATE REFINEMENT #
nnx = 50
nny = 50
he = L[0]/float(nnx-1)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("circular_damBreak")
    domain = PlanarStraightLineGraphDomain(fileprefix="circular_damBreak")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

######################
##### BATHYMETRY #####
######################

def bathymetry_function(X):
    x = X[0]
    return x * 0.


##############################
##### INITIAL CONDITIONS #####
##############################

class water_height_at_t0(object):
    def uOfXT(self, X, t):
        # position coordinates
        x = X[0]
        y = X[1]
        # dam options
        xc = opts.dam_center[0]
        yc = opts.dam_center[1]
        r = opts.dam_radius
        # define depth
        h = opts.still_water_depth
        if ((x - xc)**2 + (y - yc)**2 <= r**2):
            h = h + opts.dam_height
        return max(h - bathymetry_function(X), 0.)


class hu_at_t0(object):
    def uOfXT(self, X, t):
        return 0.

class hv_at_t0(object):
    def uOfXT(self, X, t):
        return 0.

"""
heta and hw are needed for the hyperbolic serre-green-naghdi equations.
For initial conditions, heta -> h^2, hbeta->q(dot)grad(Z), hw -> h^2div(u)+3/2*hbeta.
It's often okay to take hbeta=0. Note that the BCs for the heta and hw should be same as h
and BCs for hbeta should be same as x_mom.
For more details see: 'Hyperbolic relaxation technique for solving the dispersive Serre--Green-Naghdi Equations
with topography' by Guermond, Kees, Popov, Tovar.
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
"""Since we are using all reflecting BCs, no need to specify dirichlet conditions"""
boundaryConditions = {'water_height': lambda x, flag: None,
                      'x_mom': lambda x, flag: None,
                      'y_mom': lambda x, flag: None,
                      'h_times_eta': lambda x, flag: None,
                      'h_times_w': lambda x, flag: None,
                      'h_times_beta': lambda x, flag: None}

# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #
outputStepping = SWFlowProblem.OutputStepping(
    opts.final_time, dt_output=opts.dt_output)
initialConditions = {'water_height': water_height_at_t0(),
                     'x_mom': hu_at_t0(),
                     'y_mom': hv_at_t0(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0(),
                     'h_times_beta': hbeta_at_t0()}
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
                                              bathymetry=bathymetry_function)
mySWFlowProblem.physical_parameters['mannings'] = opts.mannings
