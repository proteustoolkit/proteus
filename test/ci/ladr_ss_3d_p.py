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
        "none, embedded, robin, immersed, scifem, pg or both"),
    ("exact", "smooth", "manufactured solution: 'smooth' (sin*cos*exp) or 'quadratic' (in P2's space, so P2 must reproduce it)"),
])

name = opts.name
nd = 3
L = (1.0, 1.0, 1.0)
x0 = (0.0, 0.0, 0.0)
domain = Domain.RectangularDomain(L=L, x=x0, name="adr3d", units="m")
from proteus import MeshTools
MeshTools.getMeshOptions(domain).nLayersOfOverlapForParallel = 0

kx, ky, kz, cx = 1.3, 0.7, 0.5, 0.4
# 'smooth': pure diffusion with variable a(x), for the L2 rates. 'quadratic': advection with constant a,
# which a P2 space reproduces exactly. The two are not combined: ADR's subgrid-error term uses the
# strong residual b.grad(u) - f without the diffusion term, so with b != 0 and a variable a (or any
# non-polynomial u) it is inconsistent and the rates drop to ~1 (measured, same in any dimension).
if opts.exact == "smooth":
    bvec = numpy.zeros(3)
    ACOEF, A0 = 0.3, 1.0
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

ans = Ans() if opts.exact == "smooth" else QuadAns()
lapOfU = (lambda x: -kx**2 - ky**2 + kz**2) if opts.exact == "smooth" else (lambda x: 2.0 + 4.0 - 2.0)

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
    return d - 0.3, n

def interface_u(x, t):
    return ans.uOfX(x)

cutOptions = {"none": {},
              "embedded": dict(embeddedBoundary=True),
              "robin": dict(embeddedBoundary=True, embeddedBoundary_robin=True),
              "immersed": dict(immersedBoundary=True),
              "scifem": dict(immersedBoundary=True, immersedSCIFEM_switch=1.0),
              "pg": dict(immersedBoundary=True, PG=True),
              "both": dict(embeddedBoundary=True, immersedBoundary=True)}[opts.cut]
if cutOptions:
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
