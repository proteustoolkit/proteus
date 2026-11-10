"""NCLS's TwoPhaseFlow physics without a boundary-condition module and without ``ncls_DBC``.

That branch of ``ParametersModelNCLS._initializePhysics`` meant to set ``p.dirichletConditions = {0: lambda x, t: None}`` (no
Dirichlet value anywhere). It assigned ``p.dirichletCondtions`` (a typo), so the p module's ``dirichletConditions`` stayed
at what it was before (None by default). The test runs ``_initializePhysics`` on that branch with a stubbed problem.
"""
from unittest import mock

import proteus.TwoPhaseFlow.utils.Parameters as P


def run_branch():
    problem = mock.MagicMock()
    problem.domain.useSpatialTools = False
    problem.SystemPhysics.boundaryConditions = {}      # no 'ncls_DBC'
    problem.SystemPhysics._modelIdxDict = {'ncls': 2, 'rans2p': 0, 'rdls': 3}
    m = P.ParametersModelNCLS(problem)
    m.p.dirichletConditions = None
    m.p.coefficients = mock.MagicMock()
    m._initializePhysics()
    return m.p


def test_dirichlet_conditions_set_without_bc():
    p = run_branch()
    assert p.dirichletConditions is not None and list(p.dirichletConditions) == [0]
    assert p.dirichletConditions[0]((0.0, 0.0, 0.0), 0.0) is None     # no Dirichlet value at any point
    assert p.diffusiveFluxBoundaryConditions == {0: {}}
