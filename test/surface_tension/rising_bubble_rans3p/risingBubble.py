from math import *
import proteus.MeshTools
from proteus import Domain
from proteus.default_n import *
from proteus.Profiling import logEvent
try:
    from .parameters import *
except:
    from parameters import *

test_case=1
AUTOMATED_TEST = True
rdls_copyList=False
# ----- PARAMETERS FOR ELLIPTIC REDISTANCING ----- #
EXPLICIT_VOF=True
EXPLICIT_NCLS=True
ELLIPTIC_REDISTANCING=2
alpha_REDISTANCING='inf' #1.0E6

# ----- PARAMETERS FOR STABILIZATION OF NS ----- #
# see parameters.py

# ----- DIMENSIONS AND REFINEMENT ----- #
nd=ct.nd
structured = True
if AUTOMATED_TEST==True:
    Refinement = 1
else:
    Refinement = 2

# ----- PHYSICAL PARAMETERS ----- #
if nd==3:
    T=1.0
    # Water
    rho_0 = 998.2
    nu_0 = 1.004e-6
    # Air
    rho_1 = 1.205
    nu_1 = 1.500e-5
    # Surface tension
    sigma_01 = 72.8
    # Gravity
    g = [0.0, 0.0, -9.8]
else:
    if test_case==1:
        T=3.0
        # Water
        rho_0 = 1000.0
        nu_0 = 10.0/rho_0
        # Air
        rho_1 = 100.0
        nu_1 = 1.0/rho_1
        # Surface tension
        sigma_01 = 24.5
        # Gravity
        g = [0.0, -0.98]
    elif test_case==2:
        T=3.0
        # Water
        rho_0 = 1000.0
        nu_0 = 10.0/rho_0
        # Air
        rho_1 = 1.0
        nu_1 = 0.1/rho_1
        # Surface tension
        sigma_01 = 1.96 
        # Gravity
        g = [0.0, -0.98]
    else:
        T=1.25
        # Water
        rho_0 = 998.2
        nu_0 = 1.004e-6
        # Air
        rho_1 = 1.205
        nu_1 = 1.500e-5
        # Surface tension
        sigma_01 = 72.8
        # Gravity
        g = [0.0, -9.8]
    
# ----- Discretization -- input options ----- #
genMesh = False#True
movingDomain = False
applyRedistancing = True
useOldPETSc = False
useSuperlu = True
timeDiscretization = 'vbdf'#vbdf'#'vbdf'  # 'vbdf', 'be', 'flcbdf'
spaceOrder = 2
pspaceOrder = 1
useHex = False
useRBLES = 0.0
useMetrics = 1.0
applyCorrection = True
useVF = 0.0
useOnlyVF = False
openTop=True

# Input checks
if spaceOrder not in [1, 2]:
    print("INVALID: spaceOrder" + spaceOrder)
    sys.exit()

if useRBLES not in [0.0, 1.0]:
    print("INVALID: useRBLES" + useRBLES)
    sys.exit()

if useMetrics not in [0.0, 1.0]:
    print("INVALID: useMetrics")
    sys.exit()

if spaceOrder == 1:
    hFactor = 1.0
    if useHex:
        quad=True
        basis = C0_AffineLinearOnCubeWithNodalBasis
        elementQuadrature = CubeGaussQuadrature(nd, 2)
        elementBoundaryQuadrature = CubeGaussQuadrature(nd - 1, 2)
    else:
        basis = C0_AffineLinearOnSimplexWithNodalBasis
        elementQuadrature = SimplexGaussQuadrature(nd, 3)
        elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 3)
elif spaceOrder == 2:
    hFactor = 0.5
    if useHex:
        quad=True
        basis = C0_AffineLagrangeOnCubeWithNodalBasis
        elementQuadrature = CubeGaussQuadrature(nd, 4)
        elementBoundaryQuadrature = CubeGaussQuadrature(nd - 1, 4)
    else:
        basis = C0_AffineQuadraticOnSimplexWithNodalBasis
        elementQuadrature = SimplexGaussQuadrature(nd, 5)
        elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 5)

if pspaceOrder == 1:
    if useHex:
        pbasis = C0_AffineLinearOnCubeWithNodalBasis
    else:
        pbasis = C0_AffineLinearOnSimplexWithNodalBasis
elif pspaceOrder == 2:
    if useHex:
        pbasis = C0_AffineLagrangeOnCubeWithNodalBasis
    else:
        pbasis = C0_AffineQuadraticOnSimplexWithNodalBasis

if AUTOMATED_TEST==True:
    T=ct.T
# Domain and mesh
if nd==2:
    L = (1.0 , 2.0)
    he = L[0]/float(4*Refinement-1)
    he*=0.5
    he*=0.5
else:
    L = (1.0 , 1.0, 2.0)
    he = L[0]/float(4*Refinement-1)
    if AUTOMATED_TEST==False:
        he*=0.65
            
weak_bc_penalty_constant = 1.0E6
nLevels = 1
#parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.element
parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0

if useHex:
    boundaries = ['left', 'right', 'bottom', 'top', 'front', 'back']
    boundaryTags = dict([(key, i + 1) for (i, key) in enumerate(boundaries)])
    nnx = 2 * Refinement + 1
    if nd==2:
        nny = 4 * Refinement + 1
    else:
        nny = 2 * Refinement + 1
        nnz = 4 * Refinement + 1
    hex = True
    domain = Domain.RectangularDomain(L)
else:
    boundaries = ['bottom', 'right', 'top', 'left', 'front', 'back']
    boundaryTags = dict([(key, i + 1) for (i, key) in enumerate(boundaries)])
    if structured:
        nnx = 5#4 * Refinement**2 + 1
        if nd==2:
            nny = 2*nnx
        else:
            nny = nnx
            nnz = nnx#2*nnx
        triangleFlag=1
        domain = Domain.RectangularDomain(L)
        domain.polyfile = os.path.dirname(os.path.abspath(__file__))+"/"+"mesh3D"
        he = L[0]/(nnx-1)
    else:
        if nd==2:
            vertices = [[0.0, 0.0],  #0
                        [L[0], 0.0],  #1
                        [L[0], L[1]],  #2
                        [0.0, L[1]]]  #3
            vertexFlags = [boundaryTags['bottom'],
                           boundaryTags['bottom'],
                           boundaryTags['top'],
                           boundaryTags['top']]
            segments = [[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 0]]
            segmentFlags = [boundaryTags['bottom'],
                            boundaryTags['right'],
                            boundaryTags['top'],
                            boundaryTags['left']]
            regions = [[1.2, 0.6]]
            regionFlags = [1]
            domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                                          vertexFlags=vertexFlags,
                                                          segments=segments,
                                                          segmentFlags=segmentFlags,
                                                          regions=regions,
                                                          regionFlags=regionFlags)
        else:
            vertices=[[0.0,0.0,0.0],#0
                      [L[0],0.0,0.0],#1
                      [L[0],L[1],0.0],#2
                      [0.0,L[1],0.0],#3
                      [0.0,0.0,L[2]],#4
                      [L[0],0.0,L[2]],#5
                      [L[0],L[1],L[2]],#6
                      [0.0,L[1],L[2]]]#7
            vertexFlags=[boundaryTags['left'],
                         boundaryTags['right'],
                         boundaryTags['right'],
                         boundaryTags['left'],
                         boundaryTags['left'],
                         boundaryTags['right'],
                         boundaryTags['right'],
                         boundaryTags['left']]
            facets=[[[0,1,2,3]],
                    [[0,1,5,4]],
                    [[1,2,6,5]],
                    [[2,3,7,6]],
                    [[3,0,4,7]],
                    [[4,5,6,7]]]
            facetFlags=[boundaryTags['bottom'],
                        boundaryTags['front'],
                        boundaryTags['right'],
                        boundaryTags['back'],
                        boundaryTags['left'],
                        boundaryTags['top']]
            regions=[[0.5*L[0],0.5*L[1],0.5*L[2]]]
            regionFlags=[1]
            domain = Domain.PiecewiseLinearComplexDomain(vertices=vertices,
                                                         vertexFlags=vertexFlags,
                                                         facets=facets,
                                                         facetFlags=facetFlags,
                                                         regions=regions,
                                                         regionFlags=regionFlags)            
        #go ahead and add a boundary tags member
        domain.boundaryTags = boundaryTags
        domain.writePoly("mesh")
        domain.writePLY("mesh")
        domain.writeAsymptote("mesh")
        if nd==2:
            triangleOptions = "VApq30Dena%8.8f" % ((he**2)/2.0,)
        else:
            triangleOptions="VApq1.4q12feena%21.16e" % ((he**3)/6.0,)
            
        logEvent("""Mesh generated using: tetgen -%s %s""" % (triangleOptions, domain.polyfile + ".poly"))

domain.MeshOptions.nnx = nnx 
domain.MeshOptions.nny = nny
domain.MeshOptions.nnz = nnz
# Numerical parameters
ns_forceStrongDirichlet = False
ns_sed_forceStrongDirichlet = False
if useMetrics:
    ns_shockCapturingFactor  = 0.5
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ls_shockCapturingFactor  = 0.5
    ls_lag_shockCapturing = True
    ls_sc_uref  = 1.0
    ls_sc_beta  = 1.0
    vof_shockCapturingFactor = 0.5
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    rd_shockCapturingFactor  = 0.5
    rd_lag_shockCapturing = False
    epsFact_density    = 1.5
    epsFact_viscosity  = epsFact_curvature  = epsFact_vof = epsFact_consrv_heaviside = epsFact_consrv_dirac = epsFact_density
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 10.0
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
    ns_shockCapturingFactor = 0.9
    ns_lag_shockCapturing = True
    ns_lag_subgridError = True
    ns_sed_shockCapturingFactor = 0.9
    ns_sed_lag_shockCapturing = True
    ns_sed_lag_subgridError = True
    ls_shockCapturingFactor = 0.9
    ls_lag_shockCapturing = True
    ls_sc_uref = 1.0
    ls_sc_beta = 1.0
    vof_shockCapturingFactor = 0.9
    vof_lag_shockCapturing = True
    vof_sc_uref = 1.0
    vof_sc_beta = 1.0
    vos_shockCapturingFactor = 0.9
    vos_lag_shockCapturing = True
    vos_sc_uref = 1.0
    vos_sc_beta = 1.0
    rd_shockCapturingFactor = 0.9
    rd_lag_shockCapturing = False
    epsFact_density = 1.5
    epsFact_viscosity = epsFact_curvature = epsFact_vof = epsFact_vos = epsFact_consrv_heaviside = epsFact_consrv_dirac = \
        epsFact_density
    epsFact_redistance = 0.33
    epsFact_consrv_diffusion = 0.1
    redist_Newton = False
    kappa_shockCapturingFactor = 0.9
    kappa_lag_shockCapturing = True  #False
    kappa_sc_uref = 1.0
    kappa_sc_beta = 1.0
    dissipation_shockCapturingFactor = 0.9
    dissipation_lag_shockCapturing = True  #False
    dissipation_sc_uref = 1.0
    dissipation_sc_beta = 1.0

ns_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
ns_sed_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
vof_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
vos_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
ls_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
rd_nl_atol_res = max(1.0e-10, 0.05 * he)
mcorr_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
kappa_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
dissipation_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
phi_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)
pressure_nl_atol_res = max(1.0e-10, 0.01 * he ** 2)

#turbulence
ns_closure = 0  #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega
ns_sed_closure = 0  #1-classic smagorinsky, 2-dynamic smagorinsky, 3 -- k-epsilon, 4 -- k-omega
    
# Sediment
rho_s = rho_0
nu_s = 10000.0*nu_0
dragAlpha = 0.0

# Time stepping
dt_fixed = 0.01#0.03
#dt_init = dt_fixed
dt_init = min(0.1*dt_fixed,0.001)
runCFL=0.33
nDTout = int(round(T/dt_fixed))

##########################################
#            Signed Distance             #
##########################################
def signedDistanceToBubble(x):
    #center of bubble
    xB = 0.5
    yB = 0.5
    zB = 0.5
    rB = 0.25
    # dist to center of bubble
    if nd==2:
        r = np.sqrt((x[0]-xB)**2 + (x[1]-yB)**2)
    else:
        r = np.sqrt((x[0]-xB)**2 + (x[1]-yB)**2 + (x[2]-zB)**2)
    # dist to surface of bubble
    dB = rB - r
    return dB
