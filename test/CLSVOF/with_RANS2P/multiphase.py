"""
Multiphase Water Test
"""
import numpy as np
from math import sqrt
from proteus import (Domain, Context,
                     FemTools as ft,
                     MeshTools as mt,
                     WaveTools as wt)
from proteus import SpatialTools as st
from proteus.Profiling import logEvent
from proteus.SpatialTools import Tank2D

AUTOMATED_TEST=True

# ----- PARAMETERS FOR CLSVOF ----- #
useCLSVOF=True
epsFactHeaviside_clsvof=1.5
lambdaFact_clsvof=10.0
computeMetrics_clsvof=0
eps_tolerance_clsvof=False
#clsvof_nl_atol_res #see below
CLSVOF_model=None

# ----- REFINEMENT ----- #
refinement = 3
structured=True

# tank
tank_dim = (1.0,1.0)

# ----- MODELS ----- #
if useCLSVOF:
    CLSVOF_model=1
    VF_model=None
    LS_model=None
else:
    CLSVOF_model=None
    VF_model=1
    LS_model=2

##########################################
#     Discretization Input Options       #
##########################################

backgroundDiffusionFactor = 0.01
genMesh = True
movingDomain = False
applyRedistancing = True
useOldPETSc = False
useSuperlu = True
timeDiscretization = 'be'  # 'vbdf', 'be', 'flcbdf'
spaceOrder = 1
useHex = False
useRBLES = 0.0
useMetrics = 1.0
applyCorrection = True
useVF = 1.0
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
# Numerical Options and Other Parameters #
##########################################

# parallel
parallelPartitioningType = mt.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0

#
weak_bc_penalty_constant = 1.0E6
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
g = [0.0, -9.8, 0.0]

# ----- TIME STEPPING & VELOCITY----- #
if AUTOMATED_TEST:
    T=0.1
    dt_fixed=0.1
else:
    T=3.0
    dt_fixed = 0.01
dt_init = min(0.1 * dt_fixed, 0.001)
runCFL = 0.33
nDTout = int(round(T/dt_fixed))

# ----- DOMAIN ----- #

if useHex:
    raise("Not implemented")
elif structured:
    nnx = 4 * refinement**2 +2
    nny = nnx
    domain = Domain.RectangularDomain(tank_dim)
    boundaryTags = domain.boundaryTags
else:
    domain = Domain.PlanarStraightLineGraphDomain()
triangleFlag=1
# ----- TANK ----- #
tank = Tank2D(domain, tank_dim)

# ----- GAUGES ----- #

# ----- EXTRA BOUNDARY CONDITIONS ----- #

tank.BC['y+'].setAtmosphere()
tank.BC['y-'].setFreeSlip()
tank.BC['x+'].setFreeSlip()
tank.BC['x-'].setFreeSlip()

# ----- MESH CONSTRUCTION ----- #

he = tank_dim[0]/float(4*refinement-1)
domain.MeshOptions.he = he
st.assembleDomain(domain)
triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)

# ----- STRONG DIRICHLET ----- #

ns_forceStrongDirichlet = False

# ----- NUMERICAL PARAMETERS ----- #
if useMetrics:
    ns_shockCapturingFactor  = 0.25
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor  = 0.25
    ls_lag_shockCapturing = True
    ls_sc_uref  = 1.0
    ls_sc_beta  = 1.0
    vof_shockCapturingFactor = 0.25
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor  = 0.25
    rd_lag_shockCapturing = False
    epsFact_density = epsFact_viscosity = epsFact_curvature \
        = epsFact_vof = ecH \
        = epsFact_consrv_dirac = epsFact_density \
        = 3.0
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 0.1
    redist_Newton = True
    kappa_shockCapturingFactor = 0.25
    kappa_lag_shockCapturing = True#False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.25
    dissipation_lag_shockCapturing = True#False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0
else:
    ns_shockCapturingFactor  = 0.9
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor  = 0.9
    ls_lag_shockCapturing = True
    ls_sc_uref  = 1.0
    ls_sc_beta  = 1.0
    vof_shockCapturingFactor = 0.9
    vof_lag_shockCapturing = True
    vof_sc_uref  = 1.0
    vof_sc_beta  = 1.0
    rd_shockCapturingFactor  = 0.9
    rd_lag_shockCapturing = False
    epsFact_density = epsFact_viscosity = epsFact_curvature \
        = epsFact_vof = ecH \
        = epsFact_consrv_dirac = epsFact_density \
        = 1.5
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 1.0
    redist_Newton = False
    kappa_shockCapturingFactor = 0.9
    kappa_lag_shockCapturing = True#False
    kappa_sc_uref  = 1.0
    kappa_sc_beta  = 1.0
    dissipation_shockCapturingFactor = 0.9
    dissipation_lag_shockCapturing = True#False
    dissipation_sc_uref  = 1.0
    dissipation_sc_beta  = 1.0

# ----- NUMERICS: TOLERANCES ----- #

ns_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
vof_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
ls_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
rd_nl_atol_res = max(1.0e-8, 0.005 * he)
mcorr_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
clsvof_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
kappa_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)
dissipation_nl_atol_res = max(1.0e-8, 0.001 * he ** 2)

# ----- TURBULENCE MODELS ----- #

ns_closure=0 #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega
if useRANS == 1:
    ns_closure = 3
elif useRANS == 2:
    ns_closure = 4

##########################################
#            Signed Distance             #
##########################################
def signedDistance(x):
    xB = 0.5
    yB = 0.5
    rB = 0.25
    zB = 0.0
    # dist to center of bubble
    r = np.sqrt((x[0]-xB)**2 + (x[1]-yB)**2)
    # dist to surface of bubble
    dB = -(rB - r)
    return dB
