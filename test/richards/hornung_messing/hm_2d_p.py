"""Hornung-Messing manufactured problem on the unit square (no gravity).

    s(p)   = pi^2/2 - 2 atan(p)^2,  eta(p) = 2/(1+p^2)   for p < 0
    s(p)   = pi^2/2,                eta(p) = 2           for p >= 0
    p_ex   = -(x-y-t)/2                                  for x-y-t < 0
    p_ex   = -tan((e^(x-y-t)-1)/(e^(x-y-t)+1))           for x-y-t >= 0

Dirichlet data p_ex on the whole boundary, p(.,0) = p_ex(.,0), T = 0.05.
Water-content bounds [0, pi^2/2]. Used as the accuracy test of Ait Hammou
Oulhaj, Cances & Chainais-Hillairet (2018) and of Dielmann, Kuzmin & Kees (JNMA).

    parun hm_2d_p.py hm_2d_n.py -l 5 -v -C "STABILIZATION_TYPE=0 FCT=False nn=17 nDT=13"
"""
from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
from proteus import Context

ct = Context.Options([
    ("STABILIZATION_TYPE", 0, "0: Galerkin, 2: edge-based low order (with FCT=True: WRR FCT)"),
    ("FCT", False, "apply the WRR flux-corrected-transport step"),
    ("nn", 9, "nodes per side, h = 1/(nn-1)"),
    ("nDT", 4, "number of fixed time steps on [0, T]"),
    ("T", 0.05, "final time"),
    ("theta_time", 1.0, "ThetaScheme weight: 1 = backward Euler, 0.5 = Crank-Nicolson"),
    ("fct_sat_lo", True, "FCT: no correction at DOFs with a saturated low-order predictor"),
    ("strong_bc", True, "strong Dirichlet (bc_mask = 0 on the boundary), as in the JNMA formulation"),
    ("fct_type", "semi-implicit", "FCT=True: 'semi-implicit' (WRR) or 'implicit' (JNMA)"),
    ("fct_prelimiter", 3.0, "implicit FCT: nodal-variation prelimiter exponent (<= 0: off)"),
    ("beta", 0.0, "compressibility (m = rho exp(beta p) theta); the exact solution assumes 0"),
    ("par", False, "PETSc KSP linear solver and element partitioning (for mpiexec -n N parun -p ...)"),
    ("par_part", "element", "par: mesh partitioning, element or node"),
    ("par_overlap", 1, "par: element layers of overlap (FCT needs >= 1 so owned rows see their full stencil)"),
    ("quad", False, "Q1 on a uniform quadrilateral mesh (as in JNMA) instead of P1 on right triangles"),
    ("nl_test", "r", "Newton convergence test: 'r' (residual), 'du' (relative increment), 'r_and_du' (both)"),
    ("nl_unorm", "l2", "norm of the increment test: 'l2' (discrete), 'L2' (FE, consistent mass), 'L2_lumped'"),
    ("nl_rtol_du", 1.0e-12, "relative tolerance of the increment test (JNMA: 1e-12 in the FE L2 norm)"),
    ("nl_max_its", 50, "maximum nonlinear iterations per solve"),
], mutable=True)

nd = 2
L = (1.0, 1.0, 1.0)
domain = Domain.RectangularDomain(L[:2])
analyticalSolution = None


def p_exact(x, t):
    xi = x[0] - x[1] - t
    if xi < 0.0:
        return -0.5 * xi
    e = numpy.exp(xi)
    return -numpy.tan((e - 1.0) / (e + 1.0))


# Hornung-Messing law: thetaR = 0, thetaSR = pi^2/2, K_s = 2. alpha and n unused.
nMediaTypes = 1
alphaVGtypes = numpy.ones((nMediaTypes + 1,), 'd')
nVGtypes = 2.0 * numpy.ones((nMediaTypes + 1,), 'd')
thetaRtypes = numpy.zeros((nMediaTypes + 1,), 'd')
thetaSRtypes = 0.5 * numpy.pi**2 * numpy.ones((nMediaTypes + 1,), 'd')
KsTypes = 2.0 * numpy.ones((nMediaTypes + 1, nd), 'd')

galerkin = False  # use Richards.ThetaScheme (see the _n file)
LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=numpy.array([0.0, 0.0, 0.0]),
                                     density=1.0,
                                     beta=ct.beta,
                                     diagonal_conductivity=True,
                                     PSK_type='HornungMessing',
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=ct.FCT,
                                     MONOLITHIC=False,
                                     num_fct_iter=1,
                                     FCT_SATURATED_LOW_ORDER=ct.fct_sat_lo,
                                     FCT_TYPE=ct.fct_type,
                                     FCT_PRELIMITER_POWER=ct.fct_prelimiter,
                                     forceStrongConditions=ct.strong_bc,
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     cK=1.0,
                                     outputQuantDOFs=False)


def getDBC(x, flag):
    on = (x[0] in (0.0, L[0])) or (x[1] in (0.0, L[1]))
    if on:
        return lambda x, t: p_exact(x, t)


dirichletConditions = {0: getDBC}


class HM_IC:
    def uOfXT(self, x, t):
        return p_exact(x, 0.0)


initialConditions = {0: HM_IC()}
fluxBoundaryConditions = {0: 'noFlow'}


def noFlux(x, flag):
    return None


advectiveFluxBoundaryConditions = {0: noFlux}
diffusiveFluxBoundaryConditions = {0: {0: noFlux}}

T = ct.T
