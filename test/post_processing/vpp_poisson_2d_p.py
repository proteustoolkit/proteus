"""Heterogeneous Poisson problem on the unit square for the velocity post-processing tests.

-div(a(x) grad u) = f with a = diag(x + 1, y + 2) and u = x^2 + y^2; Dirichlet on x = 0, y = 0 and y = 1, the exact
diffusive flux on x = 1. The post-processed velocity is then -a grad u, so its exact value is known, and every element
balances its source against the flux through its faces.
"""
from proteus import FemTools, TransportCoefficients
from proteus.default_p import *
import numpy

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


nc = 1
analyticalSolution = {0: uEx()}
analyticalSolutionVelocity = {0: velEx()}
initialConditions = None
dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvFluxBC}
diffusiveFluxBoundaryConditions = {0: {0: getDiffFluxBC}}
coefficients = TransportCoefficients.PoissonEquationCoefficients({0: a_batched}, {0: f_batched}, nc, nd)
coefficients.variableNames = ['u0']
