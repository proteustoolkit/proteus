"""Fitted-mesh reference for the 3-phase Poisson tests of ladr_3p_2d_p.py.

Same geometry, coefficients and manufactured solutions as ladr_3p_2d_p.py, but on a mesh that
conforms to both interfaces: mesh edges lie on the fluid-fluid interface Gamma_f (the line
y = m*x + c) and on the polygon of the solid circle Gamma_s, whose inside is a hole in the mesh.
Every cell then lies in one phase, so standard P1/P2 elements are used, with no cut-cell
machinery (embeddedBoundary=False, immersedBoundary=False). The diffusion coefficient and the
source are evaluated at the quadrature points from their position, and Dirichlet data on the
outer boundary and on the circle is the exact solution.

  test=1.0  piecewise-linear exact solution (exactness check; exact for P1 and P2)
  test=2.0  trigonometric manufactured solution of Test 1 (convergence rate); with fluxJump=C,
            the Test 3 solution (g + C*d)/mub on the mub side, [mu du/dn] = C on Gamma_f, imposed
            by the interface load -int_{Gamma_f} C v ds (FluxJumpLevelModel below)
"""
from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math


opts = Context.Options([
    ("test", 1.0, "1.0 = piecewise-linear exactness check, 2.0 = Test 1 trigonometric solution"),
    ("refinement", 1, "mesh level: target edge length h = 1/(4*2**(refinement-1))"),
    ("qOrder", 6, "order of quadrature"),
    ("pOrder", 1, "order of the FEM basis (1 or 2)"),
    ("usePetsc", False, "use PETSc linear solvers"),
    ("name", "", "name of the run and of its .poly/.h5/.xmf files (default: ladr_fitted_2d_p<pOrder>_r<refinement>)"),
    ("mua", 1.0, "diffusion in Omega^- (below Gamma_f)"),
    ("mub", 5.0, "diffusion in Omega^+ (above Gamma_f)"),
    ("interfaceSlope", 1.0, "slope m of Gamma_f: y = m*x + c"),
    ("interfaceOffset", 1.0/3.0, "intercept c of Gamma_f"),
    ("solidRadius", 0.3, "radius of the solid circle"),
    ("solidCenterY", 1.0/3.0, "y-coordinate of the solid circle's centre (x = 0)"),
    ("fluxJump", 0.0, "test=2.0 only: constant flux jump [mu du/dn] = fluxJump across Gamma_f"),
])

nd = 2
name = opts.name or "ladr_fitted_2d_p%d_r%d" % (opts.pOrder, opts.refinement)
h_target = 1.0/(4.0*2.0**(opts.refinement - 1))

# --- interface geometry (same as ladr_3p_2d_p.py) --------------------------------------------
_m = opts.interfaceSlope
_c = opts.interfaceOffset
_L = math.sqrt(1.0 + _m*_m)

def interface_d(x):
    """Signed distance to Gamma_f. Negative = Omega^- (mua side)."""
    return (x[1] - _m*x[0] - _c)/_L

def interface_t(x):
    """Coordinate along Gamma_f."""
    return (x[0] + _m*(x[1] - _c))/_L

# --- exact solutions -------------------------------------------------------------------------
mua = opts.mua
mub = opts.mub

class PiecewiseLinear(AnalyticalSolutions.SteadyState):
    """u = -d in Omega^-, u = -d*mua/mub in Omega^+: continuous, continuous flux, f = 0."""
    def uOfX(self, x):
        d = interface_d(x)
        return -d if d <= 0.0 else -d*mua/mub

class Trig(AnalyticalSolutions.SteadyState):
    """u = g/mua below Gamma_f and (g + C d)/mub above it, g = sin(pi d) cos(pi t), C = fluxJump:
    [u] = 0, [mu du/dn] = C, f = 2 pi^2 g (the C d part is harmonic). Same as TrigMMS of
    ladr_3p_2d_p.py with test=2.1 and test21solutionJump=0."""
    def g(self, x):
        return math.sin(math.pi*interface_d(x))*math.cos(math.pi*interface_t(x))
    def uOfX(self, x):
        d = interface_d(x)
        return self.g(x)/mua if d <= 0.0 else (self.g(x) + opts.fluxJump*d)/mub

if opts.test == 1.0:
    ans = PiecewiseLinear()
elif opts.test == 2.0:
    ans = Trig()
else:
    assert False, "Unknown test %s" % opts.test
analyticalSolution = {0: ans}
initialConditions = None

def a(x):
    mu = mua if interface_d(x) <= 0.0 else mub
    return numpy.array([[mu, 0.0], [0.0, mu]])

def f(x):
    return 2.0*math.pi**2*ans.g(x) if opts.test == 2.0 else 0.0

aOfX = {0: a}
fOfX = {0: f}

# --- fitted mesh: PSLG with Gamma_f as interior segments and the circle as a hole -------------
bt = {'exterior': 1, 'solid': 2, 'interface': 3}
R = opts.solidRadius
C = (0.0, opts.solidCenterY)
vertices, vertexFlags, segments, segmentFlags = [], [], [], []

def add_vertex(p, flag):
    for i, q in enumerate(vertices):
        if abs(q[0] - p[0]) < 1.0e-12 and abs(q[1] - p[1]) < 1.0e-12:
            return i
    vertices.append([p[0], p[1]])
    vertexFlags.append(flag)
    return len(vertices) - 1

def add_polyline(p0, p1, flag, vflag):
    """Straight segment p0-p1, pre-split into pieces of length about h."""
    n = max(1, int(math.ceil(math.hypot(p1[0] - p0[0], p1[1] - p0[1])/h_target - 1.0e-9)))
    ids = [add_vertex((p0[0] + (p1[0] - p0[0])*k/n, p0[1] + (p1[1] - p0[1])*k/n),
                      vflag if 0 < k < n else None) for k in range(n + 1)]
    for k in range(n):
        segments.append([ids[k], ids[k + 1]])
        segmentFlags.append(flag)
    return ids

# where Gamma_f meets the outer square [-1,1]^2 and the circle
def on_line(x):
    return _m*x + _c
gf_pts = []
for x in (-1.0, 1.0):
    y = on_line(x)
    if -1.0 <= y <= 1.0:
        gf_pts.append((x, y))
for y in (-1.0, 1.0):
    x = (y - _c)/_m
    if -1.0 < x < 1.0:
        gf_pts.append((x, y))
gf_pts = sorted(set(gf_pts))
assert len(gf_pts) == 2, "Gamma_f must cross the square once"
# the line through the centre meets the circle at +-R along its direction (45 and 225 degrees
# for m = 1); the circle polygon below has a vertex there when its vertex count is a multiple of 8
assert abs(on_line(C[0]) - C[1]) < 1.0e-12, "Gamma_f is assumed to pass through the circle's centre"
dirx, diry = 1.0/_L, _m/_L
tp_lo = (C[0] - R*dirx, C[1] - R*diry)
tp_hi = (C[0] + R*dirx, C[1] + R*diry)

# outer square, split at the two points of Gamma_f
corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
for k in range(4):
    p0, p1 = corners[k], corners[(k + 1) % 4]
    pts = [p0] + sorted([q for q in gf_pts if min(p0[0], p1[0]) - 1e-12 <= q[0] <= max(p0[0], p1[0]) + 1e-12
                         and min(p0[1], p1[1]) - 1e-12 <= q[1] <= max(p0[1], p1[1]) + 1e-12 and q not in (p0, p1)],
                        key=lambda q: math.hypot(q[0] - p0[0], q[1] - p0[1])) + [p1]
    for j in range(len(pts) - 1):
        add_polyline(pts[j], pts[j + 1], bt['exterior'], bt['exterior'])

# circle as an N-gon, N a multiple of 8 so that the two triple points are vertices
N = 8*int(math.ceil(2.0*math.pi*R/h_target/8.0))
theta0 = math.atan2(diry, dirx)
circle_ids = [add_vertex((C[0] + R*math.cos(theta0 + 2.0*math.pi*k/N), C[1] + R*math.sin(theta0 + 2.0*math.pi*k/N)),
                         bt['solid']) for k in range(N)]
for k in range(N):
    segments.append([circle_ids[k], circle_ids[(k + 1) % N]])
    segmentFlags.append(bt['solid'])
assert math.hypot(vertices[circle_ids[0]][0] - tp_hi[0], vertices[circle_ids[0]][1] - tp_hi[1]) < 1e-12
assert math.hypot(vertices[circle_ids[N//2]][0] - tp_lo[0], vertices[circle_ids[N//2]][1] - tp_lo[1]) < 1e-12

# Gamma_f outside the circle: outer boundary point -> triple point, triple point -> outer boundary point
gf_lo, gf_hi = (gf_pts[0], gf_pts[1]) if gf_pts[0][0] < gf_pts[1][0] else (gf_pts[1], gf_pts[0])
add_polyline(gf_lo, tp_lo, bt['interface'], bt['interface'])
add_polyline(tp_hi, gf_hi, bt['interface'], bt['interface'])

# triple points and the ends of Gamma_f keep the flag of the boundary they lie on
vertexFlags = [bt['exterior'] if fl is None else fl for fl in vertexFlags]

domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices, vertexFlags=vertexFlags,
                                             segments=segments, segmentFlags=segmentFlags,
                                             holes=[[C[0], C[1]]], name=name)
domain.boundaryTags = bt
domain.writePoly(name)
domain.MeshOptions.triangleOptions = "VApq30Dena%8.8f" % (0.5*h_target**2,)

# --- coefficients and boundary conditions ----------------------------------------------------
coefficients = ADR.Coefficients(aOfX=aOfX, fOfX=fOfX, velocity=numpy.array([0.0, 0.0]), nc=1, nd=nd,
                                forceStrongDirichlet=True,
                                embeddedBoundary=False, immersedBoundary=False,
                                analyticalSolution=analyticalSolution)

def getDBC(x, flag):
    # exact solution on the outer boundary and on the circle; none on the interior Gamma_f segments
    if flag in (bt['exterior'], bt['solid']):
        return lambda x, t: ans.uOfX(x)

dirichletConditions = {0: getDBC}
fluxBoundaryConditions = {0: 'noFlow'}

def getFlux(x, flag):
    return lambda x, t: 0.0

advectiveFluxBoundaryConditions = {0: getFlux}
diffusiveFluxBoundaryConditions = {0: {0: getFlux}}

# --- flux jump: interface load on the Gamma_f mesh edges ---------------------------------------
# Integrating by parts on each phase gives a(u,v) = (f,v) - int_{Gamma_f} [mu du/dn] v ds, so the
# residual gets + int_{Gamma_f} C v ds. The load does not depend on u (the Jacobian is unchanged);
# it is assembled once, with Gauss points on each interface edge mapped into one adjacent cell.
class FluxJumpLevelModel(ADR.LevelModel):
    def __init__(self, *args, **kwargs):
        ADR.LevelModel.__init__(self, *args, **kwargs)
        self.fluxJumpLoad = None
        if opts.fluxJump != 0.0:
            mesh = self.mesh
            fs = self.u[0].femSpace
            basis = fs.referenceFiniteElement.localFunctionSpace.basis
            xg, wg = numpy.polynomial.legendre.leggauss(6)
            load = numpy.zeros(self.u[0].dof.shape, 'd')
            length = 0.0
            for ebN in range(mesh.nElementBoundaries_global):
                if mesh.elementBoundaryMaterialTypes[ebN] != bt['interface']:
                    continue
                n0, n1 = mesh.elementBoundaryNodesArray[ebN][:2]
                p0, p1 = mesh.nodeArray[n0][:2], mesh.nodeArray[n1][:2]
                le = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                length += le
                eN = mesh.elementBoundaryElementsArray[ebN, 0]
                v = [mesh.nodeArray[mesh.elementNodesArray[eN, k]][:2] for k in range(3)]
                Jac = numpy.array([[v[1][0] - v[0][0], v[2][0] - v[0][0]], [v[1][1] - v[0][1], v[2][1] - v[0][1]]])
                for s, w in zip(xg, wg):
                    xq = p0 + 0.5*(s + 1.0)*(p1 - p0)
                    xi = numpy.linalg.solve(Jac, xq - v[0])
                    xi3 = (xi[0], xi[1], 0.0)
                    for i, phi in enumerate(basis):
                        load[fs.dofMap.l2g[eN, i]] += opts.fluxJump*phi(xi3)*0.5*le*w
            exact = math.sqrt(1.0 + _m*_m)*(gf_hi[0] - gf_lo[0]) - 2.0*R
            assert abs(length - exact) < 1.0e-8, "interface edges: length %g, expected %g" % (length, exact)
            self.fluxJumpLoad = load
    def getResidual(self, u, r):
        ADR.LevelModel.getResidual(self, u, r)
        if self.fluxJumpLoad is not None:
            dirichlet = self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict if self.forceStrongConditions else {}
            for dofN in numpy.nonzero(self.fluxJumpLoad)[0]:
                if dofN not in dirichlet:
                    r[self.offset[0] + self.stride[0]*dofN] += self.fluxJumpLoad[dofN]

LevelModelType = FluxJumpLevelModel
