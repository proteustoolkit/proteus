"""
Dambreak flow - Collagrosi and Landrini (2003)
"""
import numpy as np
from math import sqrt
from proteus.default_n import *  
from proteus import (Domain, Context,
                     FemTools as ft,
                     #SpatialTools as st,
                     MeshTools as mt,
                     WaveTools as wt,
                     Gauges)
from proteus import SpatialTools as st
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D
from proteus.Gauges import PointGauges


# predefined options
opts=Context.Options([
    # water column 
    ("water_level", 0.6, "Height of water column in m"),
    ("water_width", 1.2, "Width of  water column in m"),
    # tank
    ("tank_dim", (1.8, 1.0), "Dimensions of the tank  in m"),
    #gravity 
    ("g",(0,-9.81,0), "Gravity vector in m/s^2"),
    # gauges
    ("gauge_output", True, "Produce gauge data"),
    ("gauge_location_p", (1.8, 0.1, 0), "Pressure gauge location in m"),
    # mesh refinement and timestep
    ("refinement", 32 ,"Refinement level, he = L/(4*refinement - 1), where L is the horizontal dimension"), 
    ("cfl", 0.33 ,"Target cfl"),
    # run time options
    #("T", 0.011,"Simulation time in s"),
    ("T", 0.05,"Simulation time in s"),
    ("dt_fixed", 0.01, "Fixed time step in s"),
    ("dt_init", 0.001 ,"Maximum initial time step in s"),
    ("useHex", False, "Use a hexahedral structured mesh"),
    ("structured", False, "Use a structured triangular mesh"),
    ("gen_mesh", False ,"Generate new mesh"),
    ("usePUMI", False ,"Generate new mesh"),
    ("adapt",0,"adapt"),
    ("fixedTimeStep",True,"choice of time stepping")
    ])


pressure_gauges = PointGauges(gauges=((('p',),
                                          ((1.8,0.1,0.0),
                                          (1.4,0.0,0.0)
                                          )),),
                                  fileName="pressure.csv")


# ----- CONTEXT ------ #

# water
waterLine_z = opts.water_level
waterLine_x = opts.water_width

# tank
tank_dim = opts.tank_dim

##########################################
#     Discretization Input Options       #
##########################################

#[temp] temporary location
backgroundDiffusionFactor = 0.01

refinement = opts.refinement
genMesh = opts.gen_mesh
usePUMI = opts.usePUMI
movingDomain = False
checkMass = False
useOldPETSc = False
useSuperlu = False
timeDiscretization = 'be'  # 'vbdf', 'be', 'flcbdf'
#timeDiscretization = 'vbdf'  # 'vbdf', 'be', 'flcbdf'
spaceOrder = 1
useHex = opts.useHex
structured = opts.structured
useRBLES = 0.0
useMetrics = 1.0
applyRedistancing = True
applyCorrection = False#True
useVF = 0.0
useOnlyVF = False
useRANS = 0  # 0 -- None
             # 1 -- K-Epsilon
             # 2 -- K-Omega

# ----- INPUT CHECKS ----- #
if spaceOrder not in [1,2]:
    raise ValueError("INVALID: spaceOrder(" + str(spaceOrder) + ")")

if useRBLES not in [0.0, 1.0]:
    raise ValueError("INVALID: useRBLES(" + str(useRBLES) + ")")

if useMetrics not in [0.0, 1.0]:
    raise ValueError("INVALID: useMetrics(" + str(useMetrics) + ")")

# ----- DISCRETIZATION ----- #
nd = 2
if spaceOrder == 1:
    hFactor = 1.0
    if useHex:
        basis = ft.C0_AffineLinearOnCubeWithNodalBasis
        elementQuadrature = ft.CubeGaussQuadrature(nd, 2)
        elementBoundaryQuadrature = ft.CubeGaussQuadrature(nd - 1, 2)
    else:
        basis = ft.C0_AffineLinearOnSimplexWithNodalBasis
        elementQuadrature = ft.SimplexGaussQuadrature(nd, 3)
        elementBoundaryQuadrature = ft.SimplexGaussQuadrature(nd - 1, 3)
elif spaceOrder == 2:
    hFactor = 0.5
    if useHex:
        basis = ft.C0_AffineLagrangeOnCubeWithNodalBasis
        elementQuadrature = ft.CubeGaussQuadrature(nd, 4)
        elementBoundaryQuadrature = ft.CubeGaussQuadrature(nd - 1, 4)
    else:
        basis = ft.C0_AffineQuadraticOnSimplexWithNodalBasis
        elementQuadrature = ft.SimplexGaussQuadrature(nd, 4)
        elementBoundaryQuadrature = ft.SimplexGaussQuadrature(nd - 1, 4)

##########################################
# Numerical Options and Other Parameters             #
##########################################

weak_bc_penalty_constant = 100.0
nLevels = 1

# ----- PHYSICAL PROPERTIES ----- #

# Water
rho_0 = 998.2
nu_0 = 1.004e-6

# Air
rho_1 = 1.205
nu_1 = 1.500e-5

# Surface Tension
sigma_01 = 0.0

# Gravity
g = opts.g

# ----- TIME STEPPING & VELOCITY----- #

T = opts.T
dt_fixed = opts.dt_fixed
dt_init = min(0.1 * dt_fixed, opts.dt_init)
runCFL = opts.cfl
nDTout = int(round(T / dt_fixed))
fixedTimeStep = opts.fixedTimeStep

# ----- DOMAIN ----- #

# domain = Domain.PlanarStraightLineGraphDomain()
# domain replacement
if useHex:
    nnx = 4 * refinement + 1
    nny = 2 * refinement + 1
    hex = True
    domain = Domain.RectangularDomain(tank_dim)
elif structured:
    nnx = 4 * refinement
    nny = 2 * refinement
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags

elif usePUMI and not genMesh:
    from proteus.MeshAdaptPUMI import MeshAdapt
    domain = Domain.PUMIDomain(dim=nd,manager=MeshAdapt.AdaptManager()) #initialize the domain
    he = 0.06
    domain.AdaptManager.PUMIAdapter.loadModelAndMesh(b"Reconstructed.dmg", b"Reconstructed.smb")
    modelDict = {'flow':0,'phase':2,'corrections':[3,4]}
    domain.AdaptManager.modelDict = modelDict
    domain.AdaptManager.sizeInputs = [b'pseudo']
    domain.AdaptManager.adapt = opts.adapt
    domain.AdaptManager.hmax = he
    domain.AdaptManager.hmin= he/4.0
    domain.AdaptManager.hphi= he/2.0
    domain.AdaptManager.numIterations= 5
    domain.AdaptManager.PUMIAdapter.setAdaptProperties(domain.AdaptManager)

else:
    domain = Domain.PlanarStraightLineGraphDomain()

if genMesh and usePUMI:
  from proteus.MeshAdaptPUMI import MeshAdapt
  domain.AdaptManager=MeshAdapt.AdaptManager()


# ----- TANK ----- #

tank = Tank2D(domain, tank_dim)

# ----- EXTRA BOUNDARY CONDITIONS ----- #

tank.BC['y+'].setAtmosphere()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()

# ----- MESH CONSTRUCTION ----- #

he = 0.06#tank_dim[0] / float(4 * refinement - 1)
domain.MeshOptions.he = he
st.assembleDomain(domain)

# ----- STRONG DIRICHLET ----- #

ns_forceStrongDirichlet = False

# ----- NUMERICAL PARAMETERS ----- #

if useMetrics:
    ns_shockCapturingFactor = 0.25
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor = 0.25
    ls_lag_shockCapturing = True
    ls_sc_uref = 1.0
    ls_sc_beta = 1.0
    vof_shockCapturingFactor = 0.25
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor = 0.25
    rd_lag_shockCapturing = True 
    epsFact_density = epsFact_viscosity = epsFact_curvature \
                    = epsFact_vof = ecH \
                    = epsFact_consrv_dirac = epsFact_density \
                    = 3.0
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 0.1
    redist_Newton = True
    kappa_shockCapturingFactor = 0.25
    kappa_lag_shockCapturing = False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.25
    dissipation_lag_shockCapturing = False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0
else:
    ns_shockCapturingFactor = 0.9
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor = 0.9
    ls_lag_shockCapturing = True
    ls_sc_uref = 1.0
    ls_sc_beta = 1.0
    vof_shockCapturingFactor = 0.9
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor = 0.9
    rd_lag_shockCapturing = False
    epsFact_density = epsFact_viscosity = epsFact_curvature \
        = epsFact_vof = ecH \
        = epsFact_consrv_dirac = epsFact_density \
        = 1.5
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 1.0
    redist_Newton = False
    kappa_shockCapturingFactor = 0.9
    kappa_lag_shockCapturing = True  #False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.9
    dissipation_lag_shockCapturing = True  #False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0

# ----- NUMERICS: TOLERANCES ----- #

ns_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
vof_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
ls_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
rd_nl_atol_res = max(1.0e-10, 0.005 * he)
mcorr_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
kappa_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)
dissipation_nl_atol_res = max(1.0e-10, 0.001 * he ** 2)

# ----- TURBULENCE MODELS ----- #

ns_closure = 2  #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega
if useRANS == 1:
    ns_closure = 3
elif useRANS == 2:
    ns_closure = 4

##########################################
#            Initial conditions for free-surface                     #
##########################################

def signedDistance(x):
    phi_x = x[0] - waterLine_x
    phi_z = x[1] - waterLine_z
    if phi_x < 0.0:
        if phi_z < 0.0:
            return max(phi_x, phi_z)
        else:
            return phi_z
    else:
        if phi_z < 0.0:
            return phi_x
        else:
            return sqrt(phi_x ** 2 + phi_z ** 2)