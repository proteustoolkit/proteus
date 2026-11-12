from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math

LevelModelType = ADR.LevelModel

"""
Manufactured steady advection-diffusion with a VARIABLE diffusion coefficient on the unit square:

    div(b u) - div(a(x) grad u) = f,   a = 1 + 0.5 x y,   b = (0.5, -0.3),
    u = sin(kx x + cx) cos(ky y),  Dirichlet data from u on all four sides.

Pins the convergence rates of ADR's subgrid-error stabilization: its strong residual must include the
diffusion term -div(a grad u) or the rates fall to ~1 (L2) whatever the polynomial order.
"""

opts = Context.Options([
    ("refinement", 0, "number of times to refine the mesh (nnx = 2**(refinement+2)+1)"),
    ("pOrder", 1, "polynomial order of the C0 Lagrange space (1 or 2)"),
    ("qOrder", 5, "order of quadrature"),
    ("usePetsc", False, "Use PETSc linear solvers (required in parallel)"),
    ("sge", 1.0, "1: the subgrid error's strong residual and tau include the diffusion term; 0: advection only (the old behaviour)"),
    ("name", "ladr_sgs_2d", "name of the test"),
])

name = opts.name
nd = 2
domain = Domain.RectangularDomain(L=(1.0, 1.0), x=(0.0, 0.0), name="adr_sgs", units="m")
from proteus import MeshTools
MeshTools.getMeshOptions(domain).nLayersOfOverlapForParallel = 2

kx, ky, cx = 2.3, 1.7, 0.4
bvec = numpy.array([0.5, -0.3, 0.0])

class Ans(AnalyticalSolutions.SteadyState):
    def uOfX(self, x):
        return math.sin(kx*x[0] + cx)*math.cos(ky*x[1])
    def duOfX(self, x):
        s, c = math.sin(kx*x[0] + cx), math.cos(kx*x[0] + cx)
        return numpy.array([kx*c*math.cos(ky*x[1]), -ky*s*math.sin(ky*x[1])])

ans = Ans()

def a(x):
    return (1.0 + 0.5*x[0]*x[1])*numpy.eye(2)

def f(x):
    u = ans.uOfX(x)
    du = ans.duOfX(x)
    lap = -(kx**2 + ky**2)*u
    ax = 1.0 + 0.5*x[0]*x[1]
    # div(a grad u) = a lap u + grad a . grad u
    return float(bvec[0]*du[0] + bvec[1]*du[1] - (ax*lap + 0.5*x[1]*du[0] + 0.5*x[0]*du[1]))

aOfX = {0: a}
fOfX = {0: f}
analyticalSolution = {0: ans}
initialConditions = None

coefficients = ADR.Coefficients(aOfX=aOfX, fOfX=fOfX, velocity=bvec[:2], nc=1, nd=nd,
                                forceStrongDirichlet=True, sge_diffusion=opts.sge,
                                analyticalSolution=analyticalSolution)

def getDBC(x, flag):
    if flag in [domain.boundaryTags[k] for k in ('left', 'right', 'bottom', 'top')]:
        return lambda x, t: ans.uOfX(x)

dirichletConditions = {0: getDBC}
fluxBoundaryConditions = {0: 'noFlow'}
def getFlux(x, flag):
    return lambda x, t: 0.0
advectiveFluxBoundaryConditions = {0: getFlux}
diffusiveFluxBoundaryConditions = {0: {0: getFlux}}
