from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math

LevelModelType = ADR.LevelModel

"""
Linear advection-diffusion-reaction at equilibrium in 2D.
"""

## \page Tests Test Problems 
# \ref ladr_ss_2d_p.py "Linear advection-diffusion-reaction at steady state"
#

##\ingroup test
#\file ladr_ss_2d_p.py
#
#\brief Linear advection-diffusion-reaction at equilibrium in 1D.
#\todo finish ladr_ss_2d_p.py doc

opts = Context.Options([
    ("test", 1.0, "which test to run (default is 1.0)"),
    ("unstructured", False, "use unstructured mesh (default is structured)"),
    ("skew", 0.0001, "skew the domain when using unstructured mesh"),
    ("refinement", 0, "number of times to refine the mesh (default is 0)"),
    ("qOrder", 6, "order of quadrature (default is 6)"),
    ("usePetsc", False, "Use PETSc linear solvers"),
    ("name", "ladr_ss_2d", "name of the test (default is ladr_ss_2d)"),
    ("immersedSCIFEM_switch", 0.0, "switch (0/1) for the SCIFEM interface consistency terms"),
    ("immersedSCIFEM_penalty", 0.0, "coefficient gamma of the interior-penalty stabilization"),
    ("PG", False, "Petrov-Galerkin: test with ordinary P1 hat functions instead of "
        "the enriched va/vb IFEM basis. Trial/solution side is unchanged."),
    ("interfaceSlope", 0.5, "slope m of the straight interface y = m*x + interfaceOffset "
        "used by test=13.0. Deliberately generic (not 0 or 1)."),
    ("interfaceOffset", 0.37, "intercept c of the test=13.0 interface. Non-dyadic so it "
        "never lands on a mesh edge or vertex.")
])

name = opts.name
nd = 2
L=(2.0,2.0)#,2.0)
x0 = (-1.0,-1.0)#,-1.0)
if opts.test == 12.0:
    L = (1.0,1.0)
    x0 = (0.0,0.0)
elif opts.unstructured:
    L=(2.0,2.0+opts.skew)#,2.0)#throw off rectangular domain
domainR = Domain.RectangularDomain(L=L,x=x0,name="adr",units="m")
domainR.writePoly("ladr_ss_2d_p")
domainUS = Domain.PlanarStraightLineGraphDomain("ladr_ss_2d_p")
domainUS.boundaryTags = domainR.boundaryTags
domain = domainR
if opts.unstructured:
    domain = domainUS    
# The ghost-penalty and SCIFEM face terms need two layers of overlap in parallel (see ADR.LevelModel).
domain.MeshOptions.nLayersOfOverlapForParallel = 2  # the n module's value is copied over it
a0=1.0
if opts.test == 2.0:
    a0 = 10.0
elif opts.test ==2.1:
    a0 = -3.0
b0 = 0.0
A0_1c={0:numpy.array([[a0,0],[0,a0]])}
B0_1c={0:numpy.array([0.0,b0])}
C0_1c={0:0.0}
M0_1c={0:0.0}

#ans = AnalyticalSolutions.LinearAD_SteadyState(b=B0_1c[0][1],a=A0_1c[0][0,0])
class LevequeLiExample1(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample1, self).__init__()
    def uOfX_inner(self, x):
        return 1.0
    def uOfX_outer(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        return 1.0 + math.log(2*r)
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class LevequeLiExample2(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample2, self).__init__()
    def uOfX_inner(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        return r**2
    def uOfX_outer(self, x):
        b=a0
        C=0.1
        r = (x[0]**2 + x[1]**2)**0.5
        return (1 - 1/(8*b) - 1/b)/4 + ((r**4)/2 + r**2)/b + C*math.log(2*r)/b
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class LevequeLiExample3(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample3, self).__init__()
    def uOfX_inner(self, x):
        import math
        return math.exp(x[0])*math.cos(x[1])
    def uOfX_outer(self, x):
        return 0.0
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class LevequeLiExample4(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample4, self).__init__()
    def uOfX_inner(self, x):
        return x[0]**2 - x[1]**2
    def uOfX_outer(self, x):
        return 0.0
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class LevequeLiExample4l(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample4l, self).__init__()
    def uOfX_inner(self, x):
        return x[0] - x[1]
    def uOfX_outer(self, x):
        return 0.0
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class PWC(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWC, self).__init__()
    def uOfX_inner(self, x):
        return 1.0
    def uOfX_outer(self, x):
        return 0.0
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class PWL(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWL, self).__init__()
    def uOfX_inner(self, x):
        return x[0] + x[1] + 1.0
    def uOfX_outer(self, x):
        return x[0] + x[1]
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class PWQ(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWQ, self).__init__()
    def uOfX_inner(self, x):
        return x[0]**2 + x[1]**2 + 1.0
    def uOfX_outer(self, x):
        return x[0]**2 + x[1]**2
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class PWCubic(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWCubic, self).__init__()
    def uOfX_inner(self, x):
        return x[0]**3 + x[1]**3 + 1.0
    def uOfX_outer(self, x):
        return x[0]**3 + x[1]**3
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class PWLStraight(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWLStraight, self).__init__()
        self.jump_x = 0.35#0.001
    def uOfX_inner(self, x):
        return -(x[0]-self.jump_x)
    def uOfX_outer(self, x):
        return -(x[0]-self.jump_x)/1000.0
    def uOfX(self, x):
        if x[0] <=self.jump_x:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class JiEtal14Example1(AnalyticalSolutions.SteadyState):
    def __init__(self, betaMinus=1.0, betaPlus=1000.0):
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        super(JiEtal14Example1, self).__init__()
    def uOfX_inner(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        return r**3/self.betaMinus
    def uOfX_outer(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        return r**3/self.betaPlus + (1.0/self.betaMinus-1.0/self.betaPlus)*0.5**3
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

class trigSolution(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(trigSolution, self).__init__()
    def uOfX_inner(self, x):
        return math.sin(math.pi*x[0])*math.sin(math.pi*x[1])
    def uOfX_outer(self, x):
        return self.uOfX_inner(x)
    def uOfX(self, x):
        return self.uOfX_inner(x)

class AdjeridEtal16Example5p1(AnalyticalSolutions.SteadyState):
    def __init__(self, betaMinus=1.0, betaPlus=1000.0):
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        super(AdjeridEtal16Example5p1, self).__init__()

    def _common(self, x):
        xx = x[0]
        yy = x[1]
        psi = yy**2 - xx**2 - (4.0/3.0)*yy + 4.0/9.0
        eta = 2.0/3.0 - xx - yy
        p1 = 6.0*xx**2 + 6.0*xx*yy - 4.0*xx + 3.0
        p2 = 2.0 + 3.0*xx - 3.0*yy
        return p1*math.cos(psi) + p2*math.sin(eta)

    def uOfX_outer(self, x):  # Omega+
        return self._common(x)/(3.0*self.betaPlus)

    def uOfX_inner(self, x):  # Omega-
        xx = x[0]
        yy = x[1]
        jump_term = (self.betaMinus/self.betaPlus - 1.0)*(3.0 - 8.0*xx + 12.0*xx*yy)
        return (jump_term + self._common(x))/(3.0*self.betaMinus)

    def uOfX(self, x):
        phi = x[1] - x[0] - 2.0/3.0
        if phi >= 0.0:
            return self.uOfX_outer(x)
        else:
            return self.uOfX_inner(x)

class PWLGeneric(AnalyticalSolutions.SteadyState):
    """Piecewise-linear across a straight interface at an arbitrary angle
    (test=13.0): test=11.0's PWLStraight (vertical interface x=jump_x,
    u_outer = u_inner/1000) generalized to interface y = interfaceSlope*x +
    interfaceOffset. u_outer is just u_inner scaled by betaMinus/betaPlus, with
    no distance normalization -- continuity and flux-matching hold at any slope
    regardless, since scaling a plane by a constant scales its whole gradient by
    that same constant. Exact for P1 at any slope, unlike test=11.0 (vertical)
    and test=12.0 (slope exactly 1.0), which makes it a useful regression check.
    """
    def __init__(self, slope, offset, betaMinus=1.0, betaPlus=1000.0):
        self.slope = slope
        self.offset = offset
        self.L = (1.0 + slope**2)**0.5
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        super(PWLGeneric, self).__init__()
    def phi(self, x):
        return x[1] - self.slope*x[0] - self.offset
    def interface_d(self, x):
        return self.phi(x)/self.L
    def uOfX_inner(self, x):
        return -self.phi(x)
    def uOfX_outer(self, x):
        return -self.phi(x)*self.betaMinus/self.betaPlus
    def uOfX(self, x):
        if self.phi(x) <= 0.0:
            return self.uOfX_inner(x)
        else:
            return self.uOfX_outer(x)

if opts.test == 1.0:
    ans = LevequeLiExample1()
elif opts.test == 2.0 or opts.test == 2.1:
    ans = LevequeLiExample2()
elif opts.test == 3.0:
    ans = LevequeLiExample3()
elif opts.test == 4.0:
    ans = LevequeLiExample4()
elif opts.test == 4.1:
    ans = LevequeLiExample4l()
elif opts.test == 5.0:
    ans = PWC()
elif opts.test == 6.0:
    ans = PWL()
elif opts.test == 7.0:
    ans = PWQ()
elif opts.test == 8.0:
    ans = JiEtal14Example1(betaMinus=1.0, betaPlus=1000.0)
elif opts.test == 9.0:
    ans = PWCubic()
elif opts.test == 10.0:
    ans = trigSolution()
elif opts.test == 11.0:
    ans = PWLStraight()
elif opts.test == 12.0:
    ans = AdjeridEtal16Example5p1(betaMinus=1.0, betaPlus=1000.0)
elif opts.test == 13.0:
    ans = PWLGeneric(opts.interfaceSlope, opts.interfaceOffset, betaMinus=1.0, betaPlus=1000.0)
else:
    assert False, "Unknown test %s" % opts.test

analyticalSolution = {0:ans}
initialConditions = None
# initialConditions = {0:ans}

center = (0.0,0.0)
radius = 0.5

# Interface material parameters (diffusion on each side of the interface).
# a(x) below uses these as the source of truth.
mua = 1.0   # inner (r <= radius)
mub = 1.0   # outer (r >  radius)
jf  = 0.0
if opts.test == 2.0:
    mua = radius**2 + 1  # = 1.25, inner diffusion at the interface
    mub = a0             # = 10.0
elif opts.test == 2.1:
    mua = radius**2 + 1  # = 1.25
    mub = a0             # = -3.0
elif opts.test == 8.0 or opts.test == 11.0:
    mub = 1000.0
elif opts.test == 12.0:
    mua = 1.0
    mub = 1000.0
elif opts.test == 13.0:
    mua = 1.0
    mub = 1000.0

if opts.test == 2.0 or opts.test == 2.1:
    # Leveque & Li 1994, Example 2 — inner diffusion varies with position
    def a(x):
        if (x[0]**2 + x[1]**2) <= radius**2:
            _a = x[0]**2 + x[1]**2 + 1
        else:
            _a = mub
        return numpy.array([[_a,0.0],[0.0,_a]])
    def f(x):
        return -(8*(x[0]**2 + x[1]**2) + 4)

if opts.test in [1.0, 3.0, 4.0, 4.1, 5.0, 6.0]:
    # Leveque & Li 1994, Example 1, 3 & 4, PWC, PWL
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return 0.0
elif opts.test == 7.0:
    # PWQ
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return -4.0
elif opts.test == 8.0:
    # Ji et al 2014, Example 1 — piecewise constant diffusion
    def a(x):
        if (x[0]**2 + x[1]**2) <= radius**2:
            return numpy.array([[mua,0.0],[0.0,mua]])
        else:
            return numpy.array([[mub,0.0],[0.0,mub]])
    def f(x):
        return -9.0*(x[0]**2 + x[1]**2)**0.5
elif opts.test == 9.0:
    # PWCubic
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return -6.0*(x[0] + x[1])
elif opts.test == 10.0:
    # trig solution
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return 2.0*math.pi**2*math.sin(math.pi*x[0])*math.sin(math.pi*x[1])
elif opts.test == 11.0:
    # PWLStraight
    def a(x):
        if x[0] <= ans.jump_x:
            return numpy.array([[mua,0.0],[0.0,mua]])
        else:
            return numpy.array([[mub,0.0],[0.0,mub]])
    def f(x):
        return 0.0
elif opts.test == 12.0:
    # Adjerid et al. 2016, Example 5.1
    def a(x):
        phi = x[1] - x[0] - 2.0/3.0
        if phi >= 0.0:
            aa = mub  # Omega+
        else:
            aa = mua  # Omega-
        return numpy.array([[aa,0.0],[0.0,aa]])

    def f(x):
        xx = x[0]
        yy = x[1]
        psi = yy**2 - xx**2 - (4.0/3.0)*yy + 4.0/9.0
        eta = 2.0/3.0 - xx - yy
        p1 = 6.0*xx**2 + 6.0*xx*yy - 4.0*xx + 3.0
        p2 = 2.0 + 3.0*xx - 3.0*yy

        dp1_dx = 12.0*xx + 6.0*yy - 4.0
        dp1_dy = 6.0*xx
        dpsi_dx = -2.0*xx
        dpsi_dy = 2.0*yy - 4.0/3.0

        grad_psi_sq = dpsi_dx*dpsi_dx + dpsi_dy*dpsi_dy
        dp1_dot_dpsi = dp1_dx*dpsi_dx + dp1_dy*dpsi_dy

        lap_g = (12.0 - p1*grad_psi_sq)*math.cos(psi) - 2.0*dp1_dot_dpsi*math.sin(psi) - 2.0*p2*math.sin(eta)
        return -(1.0/3.0)*lap_g
elif opts.test == 13.0:
    # both branches linear, so no source term
    def a(x):
        if ans.phi(x) <= 0.0:
            return numpy.array([[mua,0.0],[0.0,mua]])
        else:
            return numpy.array([[mub,0.0],[0.0,mub]])
    def f(x):
        return 0.0


aOfX = {0:a}; fOfX = {0:f}

def embeddedBoundary_sdf(x,t):
    xr = x[0] - center[0]
    yr = x[1] - center[1]
    r = (xr**2 + yr**2)**0.5
    if r > 1.0e-16:
        n = (xr/r,yr/r,0.)
    else:
        n = (1.0,0.0,0.0)
    sdf = r - radius
    return sdf,n

def embeddedBoundary_sdf_straight(x,t):
    return x[0]-ans.jump_x,(1.0,0.0,0.0)

def embeddedBoundary_sdf_diag(x,t):
    sdf = (x[1] - x[0] - 2.0/3.0)/2.0**0.5
    n = (1.0/2.0**0.5,-1.0/2.0**0.5,0.0)
    return sdf,n

def embeddedBoundary_sdf_generic(x,t):
    # generic-slope interface for test=13.0; sdf < 0 is Omega- (mua)
    d = ans.interface_d(x)
    n = (-opts.interfaceSlope/ans.L, 1.0/ans.L, 0.0)
    return d,n

if opts.test == 11.0:
    embeddedBoundary_sdf = embeddedBoundary_sdf_straight
elif opts.test == 12.0:
    embeddedBoundary_sdf = embeddedBoundary_sdf_diag
elif opts.test == 13.0:
    embeddedBoundary_sdf = embeddedBoundary_sdf_generic
#n = (1.0/2.0**0.5,-1.0/2.0**0.5,0.0)
#def embeddedBoundary_sdf(x,t):
#    n = (1.0/2.0**0.5,-1.0/2.0**0.5,0.0)
#    sdf = x[0] - x[1] - 1.0
#    return sdf,n

def embeddedBoundary_u(x,t):
    return ans.uOfX(x)

# Prescribed interface jump data. These used to be hardcoded in ADR.h behind
# `if (test == ...)` branches; they belong here with the rest of each test's
# definition. The kernel forms
#     [beta du/dn] = immersedBoundary_fluxJump + immersedBoundary_fluxJumpVector . n
#     [u]          = immersedBoundary_solutionJump
# and adds nothing when they are left as None.
immersedBoundary_fluxJump = None
immersedBoundary_fluxJumpVector = None
immersedBoundary_solutionJump = None

if opts.test == 1.0:
    # Leveque & Li 1994, Example 1: [beta du/dn] = 2 at r = 1/2
    immersedBoundary_fluxJump = lambda x,t: 2.0
elif opts.test == 2.0 or opts.test == 2.1:
    # Leveque & Li 1994, Example 2: singular interface source of intensity C = 0.1,
    # giving [beta du/dn] = 2C = 0.2 (kept in sync with LevequeLiExample2's C).
    immersedBoundary_fluxJump = lambda x,t: 0.2
elif opts.test == 3.0:
    immersedBoundary_fluxJumpVector = lambda x,t: (-math.exp(x[0])*math.cos(x[1]),
                                                    math.exp(x[0])*math.sin(x[1]), 0.0)
    immersedBoundary_solutionJump = lambda x,t: -math.exp(x[0])*math.cos(x[1])
elif opts.test == 4.0:
    immersedBoundary_fluxJumpVector = lambda x,t: (-2.0*x[0], 2.0*x[1], 0.0)
    immersedBoundary_solutionJump = lambda x,t: -(x[0]**2 - x[1]**2)
elif opts.test == 4.1:
    immersedBoundary_fluxJumpVector = lambda x,t: (-1.0, 1.0, 0.0)
    immersedBoundary_solutionJump = lambda x,t: -(x[0] - x[1])
elif opts.test in [5.0, 6.0, 7.0, 9.0]:
    # PWC, PWL, PWQ, PWCubic: constant unit jump in the solution
    immersedBoundary_solutionJump = lambda x,t: -1.0

""" coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,forceStrongDirichlet=False,
                                embeddedBoundary=True,
                                embeddedBoundary_sdf=embeddedBoundary_sdf,
                                embeddedBoundary_u=embeddedBoundary_u) """

coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,
                                forceStrongDirichlet=True,
                                mua=mua,
                                mub=mub,
                                jf=jf,
                                embeddedBoundary=False,  # every test here is Gamma_f only, no embedded solid
                                immersedBoundary=True,
                                immersedBoundary_sdf=embeddedBoundary_sdf,
                                immersedBoundary_u=embeddedBoundary_u,
                                immersedBoundary_penalty=0.0,
                                immersedBoundary_fluxJump=immersedBoundary_fluxJump,
                                immersedBoundary_fluxJumpVector=immersedBoundary_fluxJumpVector,
                                immersedBoundary_solutionJump=immersedBoundary_solutionJump,
                                immersedSCIFEM_switch=opts.immersedSCIFEM_switch,
                                immersedSCIFEM_penalty=opts.immersedSCIFEM_penalty,
                                PG=opts.PG,
                                analyticalSolution=analyticalSolution)

def getDBC(x,flag):
    if flag in [domain.boundaryTags['left'], domain.boundaryTags['right'], 
                domain.boundaryTags['bottom'], domain.boundaryTags['top']]:
        return lambda x,t: ans.uOfX(x) 
#    if flag in [domain.boundaryTags['left'], domain.boundaryTags['right']]:
#        return lambda x,t: ans.uOfX(x) 
    
dirichletConditions = {0:getDBC}

fluxBoundaryConditions = {0:'noFlow'}

def getFlux(x,flag):
    return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getFlux}

diffusiveFluxBoundaryConditions = {0:{0:getFlux}}
