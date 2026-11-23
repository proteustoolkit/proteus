from proteus import Domain, Context, MeshTools as mt
from proteus.kernels import (SW2DCV, GN_SW2DCV)
from proteus import SpatialTools as st
from proteus.Domain import RectangularDomain, PlanarStraightLineGraphDomain
from proteus.Profiling import logEvent
import proteus.SWFlow.SWFlowProblem as SWFlowProblem
import numpy as np

"""
This is the set-up for the 1993/1994 experiments of Beji and Battjes which
investigate the propagation of periodic waves over a submerged bar. The
original domain is D = [0, 37.3m]x[0,4m]. However, we introduce 6m (on the left) for wave generation and 12.7m (on the right) for wave absorption.
The variables are h (water height), h*u (x-direction momentum where u is
x-direction averaged velocity), h*v (y-direction momentum where v is
v-direction averaged velocity, h*eta, h*w, h*beta.
h*eta and h*w are needed for the hyperbolic serre-green-naghdi equations.
For initial conditions, heta -> h^2, hbeta->q(dot)grad(Z), hw -> h^2div(u)+3/2*hbeta. It's often okay to take hbeta=0. Note that the BCs for the heta and hw should be same as h and BCs for h*beta should be same as h*u.
For more details see: 'Hyperbolic relaxation technique for solving the dispersive Serre--Green-Naghdi Equations
with topography' by Guermond, Kees, Popov, Tovar.
"""

# *************************** #
# ***** GENERAL OPTIONS ***** #
# *************************** #
opts = Context.Options([
    # Simulation options
    ('sw_model', 1, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 20.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("cfl", 0.125, "Desired CFL restriction"),
    ("he", 0.33, "Mesh size for unstructured mesh"),
    ("reflecting_BCs", False, "Use reflecting BCs"),
    # Tank options
    ("tank_dim", (37.3, 4.), "Dimensions of the tank"),
    ("tank_centroid", (37.3 /2., 4./2.), "Centroid of the tank"),
    ("generation", True, "Generate waves at the left boundary (True/False)"),
    ("absorption", True, "Absorb waves at the right boundary (True/False)"),
    ("tank_sponge", (6., 12.7), "Length of relaxation zones zones (left, right)"),
    # Wave options
    ("still_water_depth", 0.4, "Still water height above floor"),
    ("waves", True, "Generate waves (True/False)"),
    ("wave_period", 2.02, "Period of the waves"),
    ("wave_height", 0.02, "Height of the waves"),
    # Gauges
    ("want_gauges", False, "Output for water height point gauge")
])


# ----- TANK / DOMAIN ----- #
domain = Domain.PlanarStraightLineGraphDomain()

waterLevel = opts.still_water_depth
tank_dim = opts.tank_dim
tank_sponge = opts.tank_sponge

he = opts.he
domain.he = he

tank = st.Tank2D(domain=domain, dim=tank_dim, coords=opts.tank_centroid)

# ----- GENERATION / ABSORPTION LAYERS ----- #

tank.setSponge(x_n=tank_sponge[0], x_p=tank_sponge[1])

if opts.generation:
    gen_start = tank.x0
    gen_length = tank_sponge[0]
if opts.absorption:
    abs_start = tank.x1
    abs_length = tank_sponge[1]

# ----- ASSEMBLE DOMAIN ----- #
domain.MeshOptions.he = he
st.assembleDomain(domain)

# Set domain coordaintes for BCs
X_BC_coords = (tank.x0 - opts.tank_sponge[0], tank.x1 + opts.tank_sponge[1])
Y_BC_coords = (tank.y0, tank.y1)


# ----- BATHYMETRY +  WAVE CONDITIONS ----- #
# Define some parameters
g = 9.81
h0 = opts.still_water_depth
amp = opts.wave_height
period = opts.wave_period
freq = 2. * np.pi / period
# use SGN dispersion relation to define wave number
k = np.sqrt(3. * freq**2 / (3. * g * h0 - h0**2 * freq**2))

# Define linear_waves
def h_wave(X, t):
    x = X[0] - X_BC_coords[0]
    h_wave = h0 + amp * np.cos(k * x - freq * t)
    return h_wave
def h_u_wave(X, t):
    x = X[0] - X_BC_coords[0]
    u_wave = amp / h0 * freq / k * np.sin(k * x - freq * t)
    return h_wave(X,t) * u_wave
def h_v_wave(X, t):
    x = X[0]
    return x * 0.
def h_eta_wave(X, t):
    return h_wave(X,t)**2
def h_w_wave(X, t):
    x = X[0]
    return x * 0.
def h_beta_wave(X, t):
    x = X[0]
    return x * 0.

def bathymetry_function(X):
    x = X[0]
    bath = 0. * x
    conds = [x < 6., (6. <= x) & (x <= 12.), (12. < x) & (x <= 14.),
             (14. < x) & (x <= 17.), 17. < x]
    values = [lambda x: 0.0,
              lambda x: 1. / 20. * (x - 6.),
              lambda x:  0.3,
              lambda x: 0.3 - 1. / 10. * (x - 14.),
              lambda x: 0.0]
    bath = np.piecewise(x, conds, values)
    return bath

# ----- INITIAL CONDITIONS (WAVES AT REST)  ----- #

class h_at_t0(object):
    def uOfXT(self, X, t):
        h0 = opts.still_water_depth
        return max(h0 - bathymetry_function(X), 0.)


class hu_at_t0(object):
    def uOfXT(self, x, t):
        return 0.


class hv_at_t0(object):
    def uOfXT(self, x, t):
        return 0.


class heta_at_t0(object):
    def uOfXT(self, X, t):
        h = h_at_t0().uOfXT(X, t)
        return h**2


class hw_at_t0(object):
    def uOfXT(self, X, t):
        return 0.


class hbeta_at_t0(object):
    def uOfXT(self, x, t):
        return 0.


# ----- BOUNDARY CONDITIONS  ----- #

def h_DBC(X, flag): #  D stands for dirichlet
    x = X[0]
    if (x == X_BC_coords[0]):
        return lambda X, t: h_wave(X, t)

def h_u_DBC(X, flag):
    x = X[0]
    if (x == X_BC_coords[0]):
        return lambda X, t: h_u_wave(X, t)
    elif (x == X_BC_coords[1]):
        return lambda X, t: 0.0

def h_v_DBC(X, flag):
    y = X[1]
    if (y == Y_BC_coords[0] or y == Y_BC_coords[1]):
        return lambda X, t: 0.0

def h_eta_DBC(X, flag): #  D stands for dirichlet
    x = X[0]
    if (x == X_BC_coords[0]):
        return lambda X, t: h_eta_wave(X, t)

def h_w_DBC(X, flag):
    x = X[0]
    if (x == X_BC_coords[0]):
        return lambda X, t: h_w_wave(X, t)
    elif (x == X_BC_coords[1]):
        return lambda X, t: 0.0

def h_beta_DBC(X, flag):
    x = X[0]
    if (x == X_BC_coords[0] or x == X_BC_coords[1]):
        return lambda X, t: 0.0

# ----- WAVE GAUGES  ----- #
heightPointGauges = PointGauges(gauges=((('h',), ((5.7, 0, 0),
                                                  (10.5, 0, 0),
                                                  (12.5, 0, 0),
                                                  (13.5, 0, 0),
                                                  (14.5, 0, 0),
                                                  (15.7, 0, 0),
                                                  (17.3, 0, 0))),),
                                activeTime=(0.01, opts.final_time),
                                fileName='island_wave_gauges.csv')


# ********************************** #
# ***** Create mySWFlowProblem ***** #
# ********************************** #
outputStepping = SWFlowProblem.OutputStepping(opts.final_time, dt_output=opts.dt_output)
initialConditions = {'water_height': h_at_t0(),
                     'x_mom': hu_at_t0(),
                     'y_mom': hv_at_t0(),
                     'h_times_eta': heta_at_t0(),
                     'h_times_w': hw_at_t0(),
                     'h_times_beta': hbeta_at_t0()}
boundaryConditions = {'water_height': lambda x, flag: None,
                      'x_mom': lambda x, flag: None,
                      'y_mom': h_v_DBC,
                      'h_times_eta': lambda x, flag: None,
                      'h_times_w': lambda x, flag: None,
                      'h_times_beta': lambda x, flag: None}
waveConditions = {'h': h_wave,
                 'h_u': h_u_wave,
                 'h_v': h_v_wave,
                 'h_eta': h_eta_wave,
                 'h_w': h_w_wave,
                 'h_beta': h_beta_wave}
mySWFlowProblem = SWFlowProblem.SWFlowProblem(sw_model=opts.sw_model,
                                              cfl=opts.cfl,
                                              outputStepping=outputStepping,
                                              he=he,
                                              domain=domain,
                                              initialConditions=initialConditions,
                                              boundaryConditions=boundaryConditions,
                                              reflectingBCs=opts.reflecting_BCs,
                                              bathymetry=bathymetry_function,
                                              waveConditions=waveConditions)
# For generation / absorption zones
mySWFlowProblem.physical_parameters['gen_start'] = gen_start
mySWFlowProblem.physical_parameters['gen_length'] = gen_length
mySWFlowProblem.physical_parameters['abs_start'] = abs_start
mySWFlowProblem.physical_parameters['abs_length'] = abs_length
if opts.want_gauges:
    mySWFlowProblem.auxiliaryVariables = [HeightPointGauges]