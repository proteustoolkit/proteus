#!/usr/bin/env python
"""The multilevel mesh hierarchy is nested; proteus did not exploit it.

Uniform refinement is nested in two ways:

* The refinement routines start numbering added nodes at the coarse node
  count, so coarse node ``i`` keeps its number on the fine level and
  ``fineMesh.nodeArray[:nNodes_coarse]`` is exactly ``coarseMesh.nodeArray``.
* Every node the refinement adds is the midpoint of a coarse element edge.

Together those mean each fine node is described by a pair of coarse nodes, and
that is enough to write the nodal transfer operators down directly -- a fine
node inherited from the coarse mesh takes its value, a midpoint takes the
average of its two parents.  No interpolation points, no element Jacobians, no
quadrature.  It is also *local*: the row for a fine node needs only that node's
two parents, so whichever rank owns the node can build the row, which is what
makes the operators tractable in parallel.

These tests pin both properties and check the resulting operator against the
one proteus builds geometrically today.
"""
import numpy as np
import pytest

from proteus import MeshTools

# these test the global hierarchy builder (see conftest.py), not hierarchies
# generated in pieces, which test_generated_hierarchy.py covers
pytestmark = pytest.mark.global_hierarchy


_HIERARCHIES = {
    'triangles_2d': lambda: MeshTools.MultilevelTriangularMesh(5, 5, 1, refinementLevels=3),
    'tetrahedra_3d': lambda: MeshTools.MultilevelTetrahedralMesh(3, 3, 3, refinementLevels=3),
}


#built when the fixture runs, not at collection, so that the global_hierarchy
#switch (conftest.py) is in effect
@pytest.fixture(scope='module', params=sorted(_HIERARCHIES, reverse=True))
def hierarchy(request):
    return _HIERARCHIES[request.param]()


def test_coarse_nodes_are_a_prefix_of_the_fine_nodes(hierarchy):
    """The property the whole hierarchical representation rests on."""
    for l in range(len(hierarchy.meshList) - 1):
        coarse = hierarchy.meshList[l]
        fine = hierarchy.meshList[l + 1]
        nNodes_coarse = coarse.nNodes_global
        np.testing.assert_array_equal(
            np.asarray(fine.nodeArray)[:nNodes_coarse],
            np.asarray(coarse.nodeArray))
        assert int(np.asarray(coarse.elementNodesArray).max()) < fine.nNodes_global, (
            "coarse element node numbers must also be valid fine node numbers")


def test_every_fine_node_sits_between_its_recorded_parents(hierarchy):
    """calculateNodeParents must describe every fine node exactly."""
    parents = hierarchy.calculateNodeParents()
    assert parents[0] is None, "the coarsest level has no parent level"
    for l in range(len(hierarchy.meshList) - 1):
        coarse_x = np.asarray(hierarchy.meshList[l].nodeArray)
        fine_x = np.asarray(hierarchy.meshList[l + 1].nodeArray)
        pair = parents[l + 1]
        assert pair.shape == (hierarchy.meshList[l + 1].nNodes_global, 2)
        np.testing.assert_array_equal(
            0.5 * (coarse_x[pair[:, 0]] + coarse_x[pair[:, 1]]), fine_x)

        nNodes_coarse = hierarchy.meshList[l].nNodes_global
        inherited = pair[:nNodes_coarse]
        assert (inherited[:, 0] == inherited[:, 1]).all(), (
            "nodes the fine mesh inherited must be their own parent")
        assert (pair[nNodes_coarse:, 0] != pair[nNodes_coarse:, 1]).all(), (
            "added nodes must bisect an edge between two distinct coarse nodes")


def test_nodal_prolongation_is_a_partition_of_unity(hierarchy):
    for l in range(len(hierarchy.meshList) - 1):
        rowptr, colind, nzval = hierarchy.nodalProlongation(l)
        assert len(rowptr) - 1 == hierarchy.meshList[l + 1].nNodes_global
        assert rowptr[-1] == len(nzval) == len(colind)
        np.testing.assert_allclose(np.add.reduceat(nzval, rowptr[:-1]), 1.0)
        assert colind.max() < hierarchy.meshList[l].nNodes_global


def test_hierarchical_geometry_shares_the_finest_nodes(hierarchy):
    """Coarse levels can view the finest coordinates instead of copying them."""
    saved = hierarchy.useHierarchicalGeometry()
    assert saved > 0
    finest = np.asarray(hierarchy.meshList[-1].nodeArray)
    for l in range(len(hierarchy.meshList) - 1):
        coarse = np.asarray(hierarchy.meshList[l].nodeArray)
        assert coarse.base is finest.base or coarse.base is finest, (
            "level %i should be a view of the finest node array" % l)
        assert not coarse.flags.writeable, (
            "shared coordinates must be read-only: a write would move every "
            "finer level's mesh too")
        with pytest.raises(ValueError):
            hierarchy.meshList[l].nodeArray[0, 0] = 1.0e30


def test_nodal_prolongation_matches_the_geometric_operator():
    """The combinatorial operator must equal the one proteus builds today.

    MultilevelProjectionOperators evaluates coarse basis functions at fine
    interpolation points, with element Jacobians and quadrature. For a C0
    affine-linear nodal space that has to come out the same as the two-line
    parent rule, and it does -- exactly, not to a tolerance.
    """
    import os

    from proteus import (LinearSolvers, NumericalSolution, default_n,
                         default_s, default_so, defaults)
    from proteus.iproteus import opts as runopts

    poissondir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'poisson_2d')
    p = defaults.load_physics('poisson_het_2d_p', poissondir)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', poissondir)
    p.meshfile = os.path.join(poissondir, 'square4x4')
    p.x0 = (0.0, 0.0, 0.0)
    p.L = (1.0, 1.0, 1.0)
    n.nLevels = 3
    n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
    n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    so = default_so
    so.name = p.name = 'nodal_vs_geometric'
    so.sList = [default_s]
    runopts.logLevel = 1
    runopts.verbose = False
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)

    model = ns.modelList[0]
    mlMesh = model.mlMeshSave
    transfers = model.meshTransfers

    for l in range(len(mlMesh.meshList) - 1):
        nCoarse = mlMesh.meshList[l].nNodes_global
        nFine = mlMesh.meshList[l + 1].nNodes_global

        rowptr, colind, nzval = mlMesh.nodalProlongation(l)
        combinatorial = np.zeros((nFine, nCoarse))
        for row in range(nFine):
            for k in range(rowptr[row], rowptr[row + 1]):
                combinatorial[row, colind[k]] += nzval[k]

        geometric = np.zeros((nFine, nCoarse))
        unit = np.zeros(nCoarse)
        column = np.zeros(nFine)
        for j in range(nCoarse):
            unit[:] = 0.0
            unit[j] = 1.0
            column[:] = 0.0
            transfers.prolong_bcListDict[0][l + 1].matvec(unit, column)
            geometric[:, j] = column

        np.testing.assert_array_equal(combinatorial, geometric)


@pytest.mark.parametrize('n_ranks', [1, 2, 3, 4])
def test_nodal_prolongation_assembles_in_parallel(n_ranks):
    """The parent rule gives each rank its own rows with no communication.

    Building the operator geometrically needs coarse basis functions evaluated
    at fine interpolation points, so it needs the coarse elements to be
    reachable -- which they are not, once the levels are partitioned
    independently.  The parent rule needs only two integers per fine node, so
    it works under any partitioning.

    The helper checks the operator against the *subdomain* meshes, which is
    what makes this meaningful: the parents are numbered as the hierarchy was
    built, partitioning renumbers the global nodes, and a version of this test
    that sampled everything in the hierarchy's own numbering agreed with itself
    while being unusable with a partitioned vector.
    """
    import os
    import shutil
    import subprocess
    import sys

    bindir = os.path.dirname(sys.executable)
    mpiexec = None
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            mpiexec = candidate
            break
    if mpiexec is None:
        mpiexec = shutil.which('mpiexec') or shutil.which('mpirun')
    if mpiexec is None:
        pytest.skip('no MPI launcher available')

    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'nodal_prolongation_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)

    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('NODAL_PROLONGATION_OK') == n_ranks, (
        result.stdout + result.stderr)
