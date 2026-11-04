#!/usr/bin/env python
"""Hierarchies of regular quadrilateral and hexahedral grids, and the vertex
interpolation that serves as their transfer operator.

Two things were missing for tensor-product elements.

There was no hexahedral hierarchy at all -- MultilevelHexahedralMesh.refine
raises -- and the quadrilateral one refined through sorted Python dictionaries,
which renumbers every node, so no level's nodes were a prefix of the next's and
the nodal operators declined it. A regular grid needs no refinement: each level
is generated at its own resolution, the element parents follow from the grid
spacing, and the levels share one coordinate frame by construction.

And the nodal operators named every fine node by a pair of coarse nodes, which
describes edge midpoints only. A quadrilateral's cell centre has four parents
and a hexahedron's face and body centres have four and eight, with bilinear and
trilinear weights, so the operator is now a CSR matrix with any number of
parents per row (MultilevelMesh.vertexInterpolation).

The check that matters is exactness on the functions the space contains: a Q1
space reproduces x*y and x*y*z, so interpolating them is a test of every weight
at once, whereas P1 reproduces only linears. (The existing geometric operators,
which evaluate coarse basis functions at the fine nodes, do not pass this on
quadrilaterals or hexahedra: their row sums go as low as 0.25 and 0.125.)
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import MeshTools

# these test the global hierarchy builder (see conftest.py), not hierarchies
# generated in pieces, which test_generated_hierarchy.py covers
pytestmark = pytest.mark.global_hierarchy

HIERARCHIES = {
    'quad': (lambda: MeshTools.MultilevelQuadrilateralMesh(
        3, 3, 1, x=1.0, y=2.0, Lx=2.0, Ly=3.0, refinementLevels=3), 2),
    'hex': (lambda: MeshTools.MultilevelHexahedralMesh(
        3, 3, 3, x=-3.0, y=-3.0, z=-3.0, Lx=6.0, Ly=6.0, Lz=6.0, refinementLevels=3), 3),
    'hex_anisotropic': (lambda: MeshTools.MultilevelHexahedralMesh(
        4, 3, 5, x=0.5, Lx=3.0, Ly=2.0, Lz=4.0, refinementLevels=2), 3),
}


def _fields(x, dim):
    out = {'constant': np.ones(x.shape[0]),
           'linear': 3.0*x[:, 0] - 2.0*x[:, 1] + (0.5*x[:, 2] if dim == 3 else 0.0) + 0.75,
           'bilinear': x[:, 0]*x[:, 1] + 0.3*x[:, 0] - x[:, 1]}
    if dim == 3:
        out['trilinear'] = x[:, 0]*x[:, 1]*x[:, 2] + x[:, 0]*x[:, 2] + 0.1
    return out


@pytest.mark.parametrize('kind', sorted(HIERARCHIES))
def test_levels_are_nested_regular_grids(kind):
    make, dim = HIERARCHIES[kind]
    ml = make()
    children = 2**dim
    for l in range(len(ml.meshList) - 1):
        coarse, fine = ml.meshList[l], ml.meshList[l+1]
        assert fine.nElements_global == children*coarse.nElements_global
        parents = np.asarray(ml.elementParentsArrayList[l+1])
        offsets = np.asarray(ml.elementChildrenOffsetsList[l])
        assert (np.diff(offsets) == children).all()
        # every fine element's centre lies inside its parent, which in a regular
        # grid is what being its child means
        xc = np.asarray(coarse.nodeArray)[np.asarray(coarse.elementNodesArray)][:, :, :dim]
        xf = np.asarray(fine.nodeArray)[np.asarray(fine.elementNodesArray)][:, :, :dim]
        centre = xf.mean(axis=1)
        assert ((centre >= xc[parents].min(axis=1) - 1e-12) & (centre <= xc[parents].max(axis=1) + 1e-12)).all()
        # the children are a partition of the fine elements
        assert sorted(np.asarray(ml.elementChildrenArrayList[l]).tolist()) == list(range(fine.nElements_global))
        # every coarse node is a fine node: the levels share one frame
        fineNodes = {tuple(np.round(p[:dim], 12)) for p in np.asarray(fine.nodeArray)}
        assert all(tuple(np.round(p[:dim], 12)) in fineNodes for p in np.asarray(coarse.nodeArray))


@pytest.mark.parametrize('kind', sorted(HIERARCHIES))
def test_vertex_interpolation_reproduces_the_space(kind):
    make, dim = HIERARCHIES[kind]
    ml = make()
    assert ml.hasVertexInterpolation()
    for l in range(len(ml.meshList) - 1):
        rowptr, colind, nzval = ml.vertexInterpolation()[l+1]
        xc = np.asarray(ml.meshList[l].nodeArray)
        xf = np.asarray(ml.meshList[l+1].nodeArray)
        nFine = xf.shape[0]
        assert (nzval >= 0.0).all()
        for name, f in _fields(xc, dim).items():
            exact = _fields(xf, dim)[name]
            got = np.array([np.dot(nzval[rowptr[i]:rowptr[i+1]], f[colind[rowptr[i]:rowptr[i+1]]])
                            for i in range(nFine)])
            assert np.abs(got - exact).max() < 1e-12, (
                "%s level %i: the %s field is not reproduced (error %g)"
                % (kind, l, name, np.abs(got - exact).max()))


@pytest.mark.parametrize('kind,parents', [('quad', {1, 2, 4}), ('hex', {1, 2, 4, 8})])
def test_cell_and_face_centres_have_several_parents(kind, parents):
    """A node-parent *pair* cannot describe these; the CSR has to."""
    make, dim = HIERARCHIES[kind]
    ml = make()
    rowptr = ml.vertexInterpolation()[1][0]
    assert set(np.diff(rowptr).tolist()) == parents


def test_a_rotated_grid_is_declined_rather_than_misdescribed():
    """The box construction assumes axis-aligned cells and must say so.

    A rotated grid's weights are not products of one-dimensional hats; taking
    them anyway would give an operator that is wrong without complaint, so the
    caller has to be told to fall back to the general geometric one.
    """
    ml = HIERARCHIES['quad'][0]()
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for mesh in ml.meshList:
        x = np.asarray(mesh.nodeArray)
        x[:, :2] = x[:, :2] @ R.T
    ml._vertexInterpolation = None
    ml.nodeParentsArrayList = None
    assert not ml.hasVertexInterpolation()
    with pytest.raises(ValueError, match='axis-aligned'):
        ml.vertexInterpolation()


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_distributed_vertex_prolongation_on_the_subdomain_meshes(kind, n_ranks):
    """The parallel Mat, checked against the numbering the solvers use.

    Fields are sampled on each rank's *subdomain* mesh, not on the global one,
    and ghost entries are included: sampling everything in one numbering lets a
    test agree with itself while being unusable with a partitioned vector.
    """
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vertex_prolongation_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, kind],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('VERTEX_PROLONGATION_OK') == n_ranks, result.stdout + result.stderr


def _transfer_operators(kind, geometric, monkeypatch):
    """The prolongations NumericalSolution builds for a 3-level Q1 hierarchy.

    With geometric=True the vertex route is switched off, so they are built by
    evaluating coarse basis functions at the fine interpolation points -- the
    construction that has to be used for spaces whose dofs are not vertices.
    """
    from proteus.iproteus import opts
    from proteus import (FemTools, LinearSolvers, NumericalSolution, defaults, default_n,
                         default_s, default_so)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if geometric:
        monkeypatch.setattr(FemTools, 'nodalTransfersApply', lambda *a, **k: False)
    if kind == 'hex':
        d = os.path.join(root, 'ci')
        p = defaults.load_physics('poisson_3d_p', d)
        n = defaults.load_numerics('poisson_3d_c0q1_n', d)
        n.nnx = n.nny = n.nnz = 3
        p.x0, p.L = (-3., -3., -3.), (6., 6., 6.)
    else:
        d = os.path.join(root, 'poisson_2d')
        p = defaults.load_physics('poisson_het_2d_p', d)
        n = defaults.load_numerics('poisson_het_2d_c0pk_n', d)
        p.x0, p.L, p.meshfile = (1., 2., 0.), (2., 3., 1.), None
        n.quad, n.nn = True, 3
        n.femSpaces[0] = default_n.C0_AffineLinearOnCubeWithNodalBasis
        n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 3)
        n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 3)
    n.nLevels = 3
    n.multilevelLinearSolver = n.levelLinearSolver = LinearSolvers.LU
    n.linearSmoother = None
    so = default_so
    so.name = p.name = 'transfers_%s_%s' % (kind, 'geometric' if geometric else 'vertex')
    so.sList = [default_s]
    opts.logLevel, opts.verbose = 1, False
    mt = NumericalSolution.NS_base(so, [p], [n], so.sList, opts).modelList[0].meshTransfers
    assert mt.useNodalTransfers == (not geometric)
    out = []
    for l in (1, 2):
        rp, ci, nz = mt.prolong_bcListDict[0][l].getCSRrepresentation()
        dense = np.zeros((len(rp) - 1, mt.prolong_bcListDict[0][l].shape[1]))
        for i in range(len(rp) - 1):
            dense[i, ci[rp[i]:rp[i+1]]] = nz[rp[i]:rp[i+1]]
        out.append(dense)
    return out


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_geometric_and_vertex_transfers_are_the_same_operator(kind, monkeypatch):
    """Two independent constructions of the same transfer must agree.

    The geometric one evaluates coarse basis functions at fine interpolation
    points through the element maps; the vertex one writes the weights down from
    the grid. Before the cube element-map fix the geometric operators had row
    sums of 0.25 (quad) and 0.125 (hex); now they agree to the bit.
    """
    vertex = _transfer_operators(kind, False, monkeypatch)
    geometric = _transfer_operators(kind, True, monkeypatch)
    for l, (V, G) in enumerate(zip(vertex, geometric), start=1):
        assert V.shape == G.shape
        assert np.abs(V - G).max() < 1e-12, "level %i: the two constructions differ by %g" % (l, np.abs(V - G).max())
        assert np.abs(G.sum(axis=1) - 1.0).max() < 1e-12
