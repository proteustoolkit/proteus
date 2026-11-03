"""Heterogeneous Poisson problems on the unit square for the velocity post-processing tests.

The default: -div(a(x) grad u) = f with a = diag(x + 1, y + 2) and u = x^2 + y^2; Dirichlet on x = 0, y = 0 and y = 1,
the exact diffusive flux on x = 1. The post-processed velocity is then -a grad u, so its exact value is known, and every
element balances its source against the flux through its faces.

VPP_PROBLEM=lowk: -div(a grad u) = 0 with a = 1 except in the block (1/3, 2/3)^2, where a = 1e-8 (aligned with the
mesh when nnx - 1 is a multiple of 3); u = 1 on x = 0, u = 0 on x = 1, no flux on y = 0 and y = 1. The flux through
the block's faces is of the order of the block's conductivity; the flow goes around it.
"""
import os
from proteus import FemTools, TransportCoefficients
from proteus.default_p import *
import numpy

problem = os.environ.get("VPP_PROBLEM", "smooth")
lowK = 1.0e-8

nd = 2
L = (1.0, 1.0)
domain = None
polyfile = None
name = "vpp_poisson_2d"
eps = 1.0e-8


def onDirichletBoundary(x):
    return x[0] <= eps or x[1] <= eps or x[1] >= 1.0 - eps


class uEx(object):
    def uOfX(self, x):
        return x[0] ** 2 + x[1] ** 2

    def uOfXT(self, x, t):
        return self.uOfX(x)

    def duOfX(self, x):
        return numpy.array([2.0 * x[0], 2.0 * x[1]])

    def duOfXT(self, x, t):
        return self.duOfX(x)


class velEx(object):
    def uOfX(self, x):
        return -numpy.array([(x[0] + 1.0) * 2.0 * x[0], (x[1] + 2.0) * 2.0 * x[1]])

    def uOfXT(self, x, t):
        return self.uOfX(x)


def getDBC(x, flag):
    if onDirichletBoundary(x):
        return lambda x, t: uEx().uOfX(x)


def getAdvFluxBC(x, flag):
    return None


@FemTools.batchedBC
def outwardFlux(X, t):
    # the flux -a grad u . n through x = 1, n = (1, 0)
    return -(X[:, 0] + 1.0) * 2.0 * X[:, 0]


def getDiffFluxBC(x, flag):
    if x[0] >= 1.0 - eps:
        return outwardFlux


@TransportCoefficients.batched
def a_batched(X):
    a = numpy.zeros((X.shape[0], 2, 2), 'd')
    a[:, 0, 0] = X[:, 0] + 1.0
    a[:, 1, 1] = X[:, 1] + 2.0
    return a


@TransportCoefficients.batched
def f_batched(X):
    # -div(a grad u) = -(2(x+1) + 2x) - (2(y+2) + 2y)
    return -(4.0 * X[:, 0] + 2.0) - (4.0 * X[:, 1] + 4.0)


def inBlock(X):
    return (X[:, 0] > 1.0 / 3) & (X[:, 0] < 2.0 / 3) & (X[:, 1] > 1.0 / 3) & (X[:, 1] < 2.0 / 3)


@TransportCoefficients.batched
def a_lowk(X):
    a = numpy.zeros((X.shape[0], 2, 2), 'd')
    k = numpy.where(inBlock(X), lowK, 1.0)
    a[:, 0, 0] = k
    a[:, 1, 1] = k
    return a


@TransportCoefficients.batched
def f_zero(X):
    return numpy.zeros((X.shape[0],), 'd')


def getDBC_lowk(x, flag):
    if x[0] <= eps:
        return lambda x, t: 1.0
    if x[0] >= 1.0 - eps:
        return lambda x, t: 0.0


def getDiffFluxBC_lowk(x, flag):
    if (x[1] <= eps or x[1] >= 1.0 - eps) and eps < x[0] < 1.0 - eps:
        return lambda x, t: 0.0


nc = 1
analyticalSolution = {0: uEx()}
analyticalSolutionVelocity = {0: velEx()}
initialConditions = None
dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvFluxBC}
diffusiveFluxBoundaryConditions = {0: {0: getDiffFluxBC}}
coefficients = TransportCoefficients.PoissonEquationCoefficients({0: a_batched}, {0: f_batched}, nc, nd)
if problem == "lowk":
    analyticalSolution = None
    analyticalSolutionVelocity = None
    dirichletConditions = {0: getDBC_lowk}
    diffusiveFluxBoundaryConditions = {0: {0: getDiffFluxBC_lowk}}
    coefficients = TransportCoefficients.PoissonEquationCoefficients({0: a_lowk}, {0: f_zero}, nc, nd)
coefficients.variableNames = ['u0']
