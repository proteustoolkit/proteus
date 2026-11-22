from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
import numpy as np
from proteus import (Domain, Context, MeshTools as mt)
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem

"""
We now consider the experiments of [Hsiao and Lin, 2013]
 performed at the Tainan Hydraulic Laboratory in Taiwan.
 In this is set of experiments a solitary wave over-tops a seawall.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 12.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.25, "Desired CFL restriction"),
    ("refinement", 4, "Refinement level"),
    ("reflecting_BCs",False,"Use reflecting BCs"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("mannings", 0.012, "Mannings roughness coefficient")
])

###################
# DOMAIN AND MESH #
###################
L = (15.0, 2.0)  # this is domain length in x direction and y direction
refinement = opts.refinement
rectangle = RectangularDomain(L=L)

# CREATE REFINEMENT #
nnx0 = 6
nnx = (nnx0 - 1) * (2**refinement) + 1
nny = (nnx - 1)//10 + 1
he = L[0]/float(nnx-1)
if opts.structured:
    domain = rectangle
else:
    rectangle.writePoly("seawall")
    domain = PlanarStraightLineGraphDomain(fileprefix="seawall")
    domain.MeshOptions.triangleOptions = "pAq30Dena%f" % (0.5 * opts.he**2,)
    nnx = None
    nny = None

###############################
#  CONSTANTS NEEDED FOR SETUP #
###############################
g = 9.81
h0 = 0.2  # water depth
a = 0.35  # relative amplitude
k_wavenumber = np.sqrt(3.0 * a / (4.0 * h0**3))  # wavenumber
z = np.sqrt(3.0 * a * h0) / (2.0 * h0 * np.sqrt(h0 * (1.0 + a)))  # width
c = np.sqrt(g * (1.0 + a) * h0)  # wave celerity
x0 = 5.9  # initial location of solitary wave


###############################
#   Functions defined here    #
###############################


def solitary_wave(x, t):
    sechSqd = (1.00 / np.cosh(z * (x - x0 - c * t)))**2.00
    return a * h0 * sechSqd


def bathymetry_function(X):
    #  need this shift for experimental data
    x = X[0] + 3
    #  define conditionals for bathymetry
    conds = [x < 10, (13.6 < x) & (x <= 13.9), (13.9 < x) & (x <= 13.948), \
                        (13.948 < x) & (x<= 14.045)]
    bath = [lambda x: 0 - h0, \
    lambda x: 3.6 / 20. + 0.076 / (13.9 - 13.6) * (x - 13.6) - h0, \
    lambda x: 3.6 / 20. + 0.076 - h0, \
    lambda x: 3.6 / 20. + 0.076 - (0.076 - 0.022) / (14.045 - 13.948) * (x - 13.948) - h0, \
    lambda x: 1 / 20. * (x - 10.) - h0]
    return np.piecewise(x, conds, bath)

##############################
##### INITIAL CONDITIONS #####
##############################
class water_height_at_t0(object):
    def uOfXT(self, X, t):
        eta = solitary_wave(X[0], 0)
        h = max(eta - bathymetry_function(X), 0.)
        return h


class x_mom_at_t0(object):
    def uOfXT(self, X, t):
        eta = solitary_wave(X[0], 0)
        h = max(eta - bathymetry_function(X), 0.)
        x_mom = h * c * eta / (h0 + eta)
        return x_mom


class Zero(object):
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
        eta = solitary_wave(X[0], 0)
        h = max(eta - bathymetry_function(X), 0.)
        hprime = -2.0 * z * eta * np.tanh(z * (X[0] - x0 - c * t))
        hw = h * (-c * h0 * eta * hprime / (h0 + eta)**2)
        return hw


###############################
##### BOUNDARY CONDITIONS #####
###############################
X_coords = (0.0, 15.0)

def x_mom_DBC(X, flag):
    if X[0] == X_coords[0] or X[0] == X_coords[1]:
        return lambda x, t: 0.0

# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #
outputStepping = SWFlowProblem.OutputStepping(
    opts.final_time, dt_output=opts.dt_output)
initialConditions = {'water_height': water_height_at_t0(),
                     'x_mom': x_mom_at_t0(),
                     'y_mom': Zero(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0(),
                     'h_times_beta': Zero()}
boundaryConditions = {'water_height': lambda x, flag: None,
                      'x_mom': x_mom_DBC,
                      'y_mom': lambda x, flag: lambda x, t: 0.0,
                      'h_times_eta': lambda x, flag: None,
                      'h_times_w': lambda x, flag: None,
                      'h_times_beta': x_mom_DBC}
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
