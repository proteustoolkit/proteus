from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math

LevelModelType = ADR.LevelModel

"""
Manufactured steady advection-diffusion in 3D on the unit cube, no embedded or immersed boundary:

    div(b u) - div(a(x) grad u) = f,   a(x) = 1 + 0.3 x (b = 0) or a = 1.5 (b != 0),
    u = sin(kx x + c) cos(ky y) exp(kz z)  or  a quadratic,

Dirichlet data from u on all six faces. Used by test/ADR3D to pin L2 rates for P1 and P2 on the
structured tetrahedral mesh (the C44 phase 0 case: ADR in 3D without cuts).
"""

opts = Context.Options([
    ("refinement", 0, "number of times to refine the mesh (nnx = 2**(refinement+1)+1)"),
    ("pOrder", 1, "polynomial order of the C0 Lagrange space (1 or 2)"),
    ("qOrder", 5, "order of quadrature; 5 is the 15-point tetrahedron rule ADR has kernels for"),
    ("usePetsc", False, "Use PETSc linear solvers (required in parallel)"),
    ("name", "ladr_ss_3d", "name of the test"),
    ("cut", "none", "ask for a cut-cell feature, which must stop with a clear error in 3D (C44 phase 0): "
        "none, embedded, robin, immersed, scifem, pg or both; plane is a supported embedded boundary: "
        "plane / plane_nogp are a planar solid boundary (Nitsche, with / without the ghost penalty); "
        "plane_curved / sphere_curved use the P2 level set as a quadric (embeddedBoundary_curved, P2 only, C44 phase 1b); "
        "ifem_plane (manufactured two-material solution across a plane), ifem_pwl (piecewise linear: in the "
        "IFE space, so P1 reproduces it) and ifem_sphere (radial, curved interface) use the two-sided P1 "
        "IFEM basis, immersedBoundary alone (C44 phase 2); scifem, pg and both still stop"),
    ("mua", 1.0, "diffusion on the phi_f < 0 side of an ifem_* case (the two-sided IFEM basis, C44 phase 2)"),
    ("mub", 10.0, "diffusion on the phi_f > 0 side of an ifem_* case"),
    ("scifem", 0.0, "immersedSCIFEM_switch of an ifem_* case: 1 adds the SCIFEM consistency terms on cut triangular faces"),
    ("scifemPenalty", 0.0, "immersedSCIFEM_penalty (interior penalty gamma, scaled by 1/h) of an ifem_* case"),
    ("fluxJump", 0.0, "prescribed flux jump [mu du/dn] (phi_f < 0 to phi_f > 0) of ifem_plane"),
    ("solJump", 0.0, "prescribed solution jump [u] = u(phi_f > 0 side) - u(phi_f < 0 side) of ifem_plane"),
    ("planeD", 0.37, "offset d of the plane n.x = d of the plane* cases (a small-cut stress moves it to within a hair of a vertex)"),
    ("sphereR", 0.3, "radius of the embedded sphere of the embedded / sphere_curved cases"),
    ("exact", "smooth", "manufactured solution: 'smooth' (sin*cos*exp) or 'quadratic' (in P2's space, so P2 must reproduce it)"),
])

name = opts.name
nd = 3
L = (1.0, 1.0, 1.0)
x0 = (0.0, 0.0, 0.0)
domain = Domain.RectangularDomain(L=L, x=x0, name="adr3d", units="m")
from proteus import MeshTools
MeshTools.getMeshOptions(domain).nLayersOfOverlapForParallel = 2

kx, ky, kz, cx = 1.3, 0.7, 0.5, 0.4
# 'smooth': pure diffusion with variable a(x), for the L2 rates. 'quadratic': advection with constant a,
# which a P2 space reproduces exactly. The two are not combined: ADR's subgrid-error term uses the
# strong residual b.grad(u) - f without the diffusion term, so with b != 0 and a variable a (or any
# non-polynomial u) it is inconsistent and the rates drop to ~1 (measured, same in any dimension).
if opts.exact in ("smooth", "radial"):
    bvec = numpy.zeros(3)
    ACOEF, A0 = (0.3, 1.0) if opts.exact == "smooth" else (0.0, 1.0)
else:
    bvec = numpy.array([0.5, -0.3, 0.2])
    ACOEF, A0 = 0.0, 1.5

class Ans(AnalyticalSolutions.SteadyState):
    def uOfX(self, x):
        return math.sin(kx*x[0] + cx)*math.cos(ky*x[1])*math.exp(kz*x[2])
    def duOfX(self, x):
        s, c = math.sin(kx*x[0] + cx), math.cos(kx*x[0] + cx)
        cy, sy = math.cos(ky*x[1]), math.sin(ky*x[1])
        e = math.exp(kz*x[2])
        return numpy.array([kx*c*cy*e, -ky*s*sy*e, kz*s*cy*e])

class QuadAns(AnalyticalSolutions.SteadyState):
    def uOfX(self, x):
        return x[0]**2 + 2.0*x[1]**2 - x[2]**2 + x[0]*x[1] + 0.5*x[2]
    def duOfX(self, x):
        return numpy.array([2.0*x[0] + x[1], 4.0*x[1] + x[0], -2.0*x[2] + 0.5])

class RadialAns(AnalyticalSolutions.SteadyState):
    """u = rho^2 - R^2 (rho from the cube centre): in the P2 space, and 0 on the sphere of radius R, so a
    constant Dirichlet datum there is exact and the error of a cut solid measures its geometry."""
    def uOfX(self, x):
        return (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2 - opts.sphereR**2
    def duOfX(self, x):
        return numpy.array([2.0*(x[0]-0.5), 2.0*(x[1]-0.5), 2.0*(x[2]-0.5)])

ans = {"smooth": Ans, "quadratic": QuadAns, "radial": RadialAns}[opts.exact]()
lapOfU = {"smooth": (lambda x: -kx**2 - ky**2 + kz**2), "quadratic": (lambda x: 2.0 + 4.0 - 2.0),
          "radial": (lambda x: 6.0)}[opts.exact]

def a(x):
    return (A0 + ACOEF*x[0])*numpy.eye(3)

def f(x):
    u = ans.uOfX(x)
    du = ans.duOfX(x)
    lap = lapOfU(x)*(u if opts.exact == 'smooth' else 1.0)
    # div(a grad u) = a lap u + (da/dx) u_x, da/dx = 0.3
    return float(numpy.dot(bvec, du) - ((A0 + ACOEF*x[0])*lap + ACOEF*du[0]))

aOfX = {0: a}
fOfX = {0: f}
analyticalSolution = {0: ans}
initialConditions = None

def sphere_sdf(x, t):
    d = math.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2)
    n = ((x[0]-0.5)/d, (x[1]-0.5)/d, (x[2]-0.5)/d) if d > 0.0 else (0.0, 0.0, 1.0)
    return d - opts.sphereR, n

def interface_u(x, t):
    return 0.0 if opts.exact == "radial" else ans.uOfX(x)

_pn = numpy.array([1.0, 0.6, 0.3])
_pn /= numpy.linalg.norm(_pn)

def plane_sdf(x, t):
    """Signed distance to the plane n.x = 0.37, positive on the fluid side (the solid is n.x < 0.37)."""
    return float(numpy.dot(_pn, x[:3])) - opts.planeD, tuple(_pn)

# --- two-material cases for the immersed (IFEM) interface, C44 phase 2 -------------------------------------
# phi_f < 0 is mua, phi_f > 0 is mub; u is continuous and mu du/dn is continuous across Gamma_f.
_in = numpy.array([1.0, 0.6, 0.3])
_in /= numpy.linalg.norm(_in)
_id = 0.45   # plane n.x = 0.45: not aligned with the mesh, and not through a vertex
_ic = numpy.array([0.5, 0.5, 0.5]); _iR = 0.3

def ifem_plane_sdf(x, t):
    return float(numpy.dot(_in, x[:3])) - _id, tuple(_in)

def ifem_sphere_sdf(x, t):
    r = numpy.sqrt(numpy.sum((numpy.asarray(x[:3]) - _ic)**2))
    n = tuple((numpy.asarray(x[:3]) - _ic)/r) if r > 0 else (0.0, 0.0, 1.0)
    return float(r - _iR), n

class IfemPlane(AnalyticalSolutions.SteadyState):
    """u^pm = S(x) + q(x)/mu^pm with S = sin(a.x), a.n = 0 (so dS/dn = 0) and q = (n.x - d) cos(b.x):
    continuous across the plane (q = 0 there) with mu du/dn = dq/dn on both sides."""
    a = numpy.cross(_in, [0.0, 0.0, 1.0]) * 2.0 + numpy.cross(_in, [0.0, 1.0, 0.0]) * 1.3
    b = numpy.array([1.1, -0.7, 0.9])
    def __init__(self, mua, mub):
        self.mua, self.mub = mua, mub
        self.fj, self.sj = opts.fluxJump, opts.solJump
    def S(self, x):
        return math.sin(numpy.dot(self.a, x[:3]) + 0.4)
    def dS(self, x):
        return math.cos(numpy.dot(self.a, x[:3]) + 0.4)*self.a
    def q(self, x):
        return (numpy.dot(_in, x[:3]) - _id)*math.cos(numpy.dot(self.b, x[:3]))
    def dq(self, x):
        return _in*math.cos(numpy.dot(self.b, x[:3])) - (numpy.dot(_in, x[:3]) - _id)*math.sin(numpy.dot(self.b, x[:3]))*self.b
    def lapq(self, x):
        nb = numpy.dot(_in, self.b)
        return -2.0*nb*math.sin(numpy.dot(self.b, x[:3])) - (numpy.dot(_in, x[:3]) - _id)*numpy.dot(self.b, self.b)*math.cos(numpy.dot(self.b, x[:3]))
    def uOfX_inner(self, x):
        return self.S(x) + self.q(x)/self.mua
    def uOfX_outer(self, x):
        return self.S(x) + self.q(x)/self.mub + self.fj*(numpy.dot(_in, x[:3]) - _id)/self.mub + self.sj
    def duOfX_inner(self, x):
        return self.dS(x) + self.dq(x)/self.mua
    def duOfX_outer(self, x):
        return self.dS(x) + self.dq(x)/self.mub + self.fj*_in/self.mub
    def uOfX(self, x):
        return self.uOfX_inner(x) if ifem_plane_sdf(x, 0)[0] <= 0.0 else self.uOfX_outer(x)
    def f(self, x):
        mu = self.mua if ifem_plane_sdf(x, 0)[0] <= 0.0 else self.mub
        return mu*numpy.dot(self.a, self.a)*self.S(x) - self.lapq(x)

class IfemPWL(AnalyticalSolutions.SteadyState):
    """u^pm = g0 + g.x + (c/mu^pm)(n.x - d), g.n = 0: lies in the P1 IFE space for the plane n.x = d."""
    g0, c = 0.3, 1.7
    g = numpy.array([0.4, -0.2, 0.6])
    g = g - numpy.dot(g, _in)*_in      # g.n = 0, so mu du/dn = c on both sides: the flux jump is zero
    def __init__(self, mua, mub):
        self.mua, self.mub = mua, mub
    def uOfX_inner(self, x):
        return self.g0 + numpy.dot(self.g, x[:3]) + self.c/self.mua*(numpy.dot(_in, x[:3]) - _id)
    def uOfX_outer(self, x):
        return self.g0 + numpy.dot(self.g, x[:3]) + self.c/self.mub*(numpy.dot(_in, x[:3]) - _id)
    def duOfX_inner(self, x):
        return self.g + self.c/self.mua*_in
    def duOfX_outer(self, x):
        return self.g + self.c/self.mub*_in
    def uOfX(self, x):
        return self.uOfX_inner(x) if ifem_plane_sdf(x, 0)[0] <= 0.0 else self.uOfX_outer(x)
    def f(self, x):
        return 0.0

class IfemSphere(AnalyticalSolutions.SteadyState):
    """u^- = rho^2/mua, u^+ = rho^2/mub + (1/mua - 1/mub) R^2: continuous at rho = R, mu du/drho = 2 rho on both sides."""
    def __init__(self, mua, mub):
        self.mua, self.mub = mua, mub
    def uOfX_inner(self, x):
        return numpy.sum((numpy.asarray(x[:3]) - _ic)**2)/self.mua
    def uOfX_outer(self, x):
        return numpy.sum((numpy.asarray(x[:3]) - _ic)**2)/self.mub + (1.0/self.mua - 1.0/self.mub)*_iR**2
    def duOfX_inner(self, x):
        return 2.0*(numpy.asarray(x[:3]) - _ic)/self.mua
    def duOfX_outer(self, x):
        return 2.0*(numpy.asarray(x[:3]) - _ic)/self.mub
    def uOfX(self, x):
        return self.uOfX_inner(x) if ifem_sphere_sdf(x, 0)[0] <= 0.0 else self.uOfX_outer(x)
    def f(self, x):
        return -6.0

_ifem = {"ifem_plane": (IfemPlane, ifem_plane_sdf), "ifem_pwl": (IfemPWL, ifem_plane_sdf),
         "ifem_sphere": (IfemSphere, ifem_sphere_sdf)}
if opts.cut in _ifem:
    bvec = numpy.zeros(3)
    ans = _ifem[opts.cut][0](opts.mua, opts.mub)
    _isdf = _ifem[opts.cut][1]
    def a(x):
        return (opts.mua if _isdf(x, 0)[0] <= 0.0 else opts.mub)*numpy.eye(3)
    def f(x):
        return ans.f(x)
    aOfX = {0: a}
    fOfX = {0: f}
    analyticalSolution = {0: ans}

cutOptions = {"none": {},
              "plane": dict(embeddedBoundary=True),
              "plane_nogp": dict(embeddedBoundary=True, embeddedBoundary_ghost_penalty=0.0),
              "plane_curved": dict(embeddedBoundary=True, embeddedBoundary_curved=True),
              "sphere_curved": dict(embeddedBoundary=True, embeddedBoundary_curved=True),
              "embedded": dict(embeddedBoundary=True),
              "robin": dict(embeddedBoundary=True, embeddedBoundary_robin=True),
              "immersed": dict(immersedBoundary=True),
              "scifem": dict(immersedBoundary=True, immersedSCIFEM_switch=1.0),
              "pg": dict(immersedBoundary=True, PG=True),
              "both": dict(embeddedBoundary=True, immersedBoundary=True),
              "ifem_plane": dict(immersedBoundary=True, mua=opts.mua, mub=opts.mub, immersedBoundary_penalty=0.0,
                                 immersedBoundary_fluxJump=(lambda x, t: opts.fluxJump) if opts.fluxJump != 0.0 else None,
                                 immersedBoundary_solutionJump=(lambda x, t: opts.solJump) if opts.solJump != 0.0 else None),
              "ifem_pwl": dict(immersedBoundary=True, mua=opts.mua, mub=opts.mub, immersedBoundary_penalty=0.0),
              "ifem_sphere": dict(immersedBoundary=True, mua=opts.mua, mub=opts.mub, immersedBoundary_penalty=0.0)}[opts.cut]
if opts.cut in ("plane", "plane_nogp", "plane_curved"):
    cutOptions.update(embeddedBoundary_sdf=plane_sdf, embeddedBoundary_u=interface_u)
elif opts.cut in _ifem:
    cutOptions.update(immersedBoundary_sdf=_ifem[opts.cut][1], immersedBoundary_u=interface_u,
                      immersedSCIFEM_switch=opts.scifem, immersedSCIFEM_penalty=opts.scifemPenalty)
elif cutOptions:
    cutOptions.update(embeddedBoundary_sdf=sphere_sdf, embeddedBoundary_u=interface_u,
                      immersedBoundary_sdf=sphere_sdf, immersedBoundary_u=interface_u)

coefficients = ADR.Coefficients(aOfX=aOfX, fOfX=fOfX, velocity=bvec, nc=1, nd=nd,
                                forceStrongDirichlet=True,
                                analyticalSolution=analyticalSolution,
                                **cutOptions)

def getDBC(x, flag):
    if flag in [domain.boundaryTags[k] for k in ('left', 'right', 'front', 'back', 'bottom', 'top')]:
        return lambda x, t: ans.uOfX(x)

dirichletConditions = {0: getDBC}
fluxBoundaryConditions = {0: 'noFlow'}
def getFlux(x, flag):
    return lambda x, t: 0.0
advectiveFluxBoundaryConditions = {0: getFlux}
diffusiveFluxBoundaryConditions = {0: {0: getFlux}}
