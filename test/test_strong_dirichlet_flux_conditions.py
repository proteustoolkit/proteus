"""StrongDirichlet used directly (not through StrongDirichletFactory) on a model with an advection entry.

PoissonEquationCoefficients declares advection, so the exterior flux loop visits component 0. The class used to keep
an empty fluxBoundaryConditions and raised KeyError: 0 there; it now takes the p module's dictionary, and a component
without an entry has no outflow term.
"""
import numpy as np
import pytest

from proteus import NumericalFlux

P_MODULE = '''
from proteus import FemTools, TransportCoefficients
from proteus.default_p import *
import numpy

nd = 2
L = (1.0, 1.0)
name = "strong_dirichlet_poisson"
eps = 1.0e-8


def uex(x):
    return x[0] ** 2 + x[1] ** 2


def getDBC(x, flag):
    if x[0] <= eps or x[1] <= eps or x[1] >= 1.0 - eps:
        return lambda x, t: uex(x)


def getAdvFluxBC(x, flag):
    return None


@FemTools.batchedBC
def outwardFlux(X, t):
    # -a grad u . n on x = 1 with a = diag(x + 1, y + 2), n = (1, 0)
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
    return -(4.0 * X[:, 0] + 2.0) - (4.0 * X[:, 1] + 4.0)


nc = 1
initialConditions = None
dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvFluxBC}
diffusiveFluxBoundaryConditions = {0: {0: getDiffFluxBC}}
coefficients = TransportCoefficients.PoissonEquationCoefficients({0: a_batched}, {0: f_batched}, nc, nd)
'''

N_MODULE = '''
from proteus.default_n import *
from strong_dirichlet_poisson_p import *

timeIntegration = NoIntegration
nDTout = 1
femSpaces = {0: C0_AffineQuadraticOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 4)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 4)
nnx = nny = 5
nLevels = 1
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
maxNonlinearIts = 1
fullNewtonFlag = True
nl_atol_res = 1.0e-10
tolFac = 0.0
matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
numericalFluxType = StrongDirichlet
'''


def _solve(tmp_path, monkeypatch, p_extra=""):
    from proteus import NumericalSolution, default_s, runtime
    from proteus.defaults import load_physics, load_numerics, System_base
    (tmp_path / "strong_dirichlet_poisson_p.py").write_text(P_MODULE + p_extra)
    (tmp_path / "strong_dirichlet_poisson_n.py").write_text(N_MODULE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    runtime.start()
    p = load_physics("strong_dirichlet_poisson_p", str(tmp_path))
    n = load_numerics("strong_dirichlet_poisson_n", str(tmp_path))
    so = System_base()
    so.name = p.name
    so.sList = [default_s]
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
    ns.calculateSolution("strong_dirichlet")
    return ns.modelList[0].levelModelList[-1]


def test_strong_dirichlet_with_a_diffusive_flux_condition_solves(tmp_path, monkeypatch):
    model = _solve(tmp_path, monkeypatch)
    assert model.numericalFlux.__class__ is NumericalFlux.StrongDirichlet
    x = model.u[0].femSpace.dofMap.lagrangeNodesArray
    # P2 reproduces the quadratic exact solution
    assert np.max(np.abs(model.u[0].dof - (x[:, 0] ** 2 + x[:, 1] ** 2))) < 1.0e-10


def test_strong_dirichlet_takes_the_p_module_flux_conditions(tmp_path, monkeypatch):
    model = _solve(tmp_path, monkeypatch, p_extra="\nfluxBoundaryConditions = {0: 'noFlow'}\n")
    assert model.numericalFlux.fluxBoundaryConditions == {0: 'noFlow'}


def test_the_factory_still_overrides_the_p_module():
    cls = NumericalFlux.StrongDirichletFactory({0: 'outFlow'})
    obj = cls.__new__(cls)
    obj.vt = type("vt", (), {"fluxBoundaryConditions": {0: 'noFlow'}})()
    obj.setFluxBoundaryConditions()
    assert obj.fluxBoundaryConditions == {0: 'outFlow'}
