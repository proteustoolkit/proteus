#!/usr/bin/env python
"""Partitioning with prescribed owners: no PETSc, and the same arrays as before.

partitionNodes and partitionElements used to number what each rank owns with
PETSc index sets (ISPartitioningToNumbering, ISAllGather) and to decide who owns
the elements, faces, edges or nodes that follow from the partition by passing a
bit mask from rank to rank. Given the owners, they now do neither: the
numbering is a counting sort and the masks became the rule they implemented --
an entity belongs to the lowest rank that reaches it -- computed on every rank
at once. Only the graph partitioner (MatPartitioning) still needs PETSc.

What is pinned here:

* every array the partitioner produces -- the eight offset and numbering arrays
  it returns, the old-to-new permutations, the subdomain mesh, and the global
  mesh partitionElements renumbers -- is identical, entry for entry on every
  rank, to what the mask-chain partitioner produced (2d21116f) for the same
  owners: triangles, tetrahedra, quadrilaterals and hexahedra, node and element
  partitioning, 1-8 ranks (partition_arrays_reference.json, recorded on Linux);
  with ONE deliberate exception: quadrilateral *node* partitioning, whose element
  boundaries are now owned through the boundary's own nodes (the mask chain applied the
  simplex rule, which gave an edge to a rank holding neither end point and left Q2 rows
  incomplete). Only the element-boundary arrays differ from the 2d21116f recording and
  the six `quad node` digests were re-recorded; see MULTILEVEL_MG_CAMPAIGN.md 3.19; and the six `hex element` digests at 3, 4 and 8 ranks
  (C32 round 5: the mask chain's overlap gave rank > 0 a non-edge node pair as edge 0, a row (0,0)
  in edgeNodesArray; only the edge arrays differ);
* the graph partitioner's own numbering is the prescribed-owner numbering of
  the partition it chose, so that path changed in nothing but the gather;
* a rank that owns nothing works (the old partitioner crashed): every array is
  what the same partition gives on one rank fewer, with the empty rank removed;
* with petsc4py blocked and owners prescribed, PETSc is never initialized --
  for one mesh, and for a whole multigrid solve (Q1, P1, P2, nonconforming P1,
  DG P1) on a hierarchy whose coarsest level is cut into Cartesian bricks and
  whose finer levels inherit, which matches the graph-partitioned solve. That
  needs the quadratic dof numberings PETSc-free too, which they now are;
* MeshTools.cartesianNodeOwners numbers the nodes as each rectangular-grid
  generator does, and cartesianRankGrid picks the least ghost surface.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'partition_arrays_reference.json')) as f:
    REFERENCE = json.load(f)


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(args, n_ranks, blocked=False, timeout=300):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    # CI runners have two cores; let Open MPI (4.x and 5.x/PRRTE) start more
    # ranks than that. MPICH oversubscribes by default and ignores these.
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    script = [os.path.join(HERE, 'without_petsc4py.py')] if blocked else []
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable] + script +
                            [os.path.join(HERE, args[0])] + [str(a) for a in args[1:]],
                            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    got = {}
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            got.update(item.split('=', 1) for item in line.split()[1:])
        elif line.startswith('PETSC_INITIALIZED='):
            got['petsc_initialized'] = line.split('=')[1]
    assert got, result.stdout + result.stderr
    return got


KINDS = ['tri', 'quad', 'tet', 'hex']
BY = ['node', 'element']


@pytest.mark.parametrize('n_ranks', [1, 3, 4, 8])
@pytest.mark.parametrize('owners', ['brick', 'scatter'])
@pytest.mark.parametrize('by', BY)
@pytest.mark.parametrize('kind', KINDS)
def test_arrays_identical_to_the_mask_chain_partitioner(kind, by, owners, n_ranks):
    got = _launch(['partition_arrays_mpi.py', kind, by, owners], n_ranks)
    assert got['digest'] == REFERENCE['%s %s %s %d' % (kind, by, owners, n_ranks)], got


@pytest.mark.parametrize('by', BY)
@pytest.mark.parametrize('kind', KINDS)
def test_graph_partitioner_numbers_its_partition_as_prescribed_owners_would(kind, by):
    got = _launch(['partition_arrays_mpi.py', kind, by, 'graphreplay'], 3)
    assert int(got['mismatches']) == 0, got


@pytest.mark.parametrize('n_ranks', [3, 5])
@pytest.mark.parametrize('by', BY)
@pytest.mark.parametrize('kind', KINDS)
def test_a_rank_that_owns_nothing(kind, by, n_ranks, tmp_path):
    empty = tmp_path / 'empty'
    closed = tmp_path / 'closedup'
    _launch(['partition_arrays_mpi.py', kind, by, 'empty', empty], n_ranks)
    _launch(['partition_arrays_mpi.py', kind, by, 'closedup', closed], n_ranks - 1)
    for r in range(n_ranks):
        got = np.load(empty / ('rank%d.npz' % r))
        if r == 1:
            assert int(got['subdomain.nNodes_global']) == 0
            assert int(got['subdomain.nElements_global']) == 0
            continue
        want = np.load(closed / ('rank%d.npz' % (r - 1 if r > 1 else 0)))
        assert sorted(got.files) == sorted(want.files)
        for name in want.files:
            a = got[name]
            if name.endswith('Offsets_subdomain_owned'):
                # the empty rank's entry repeats the one before it
                assert a[1] == a[2], (name, a)
                a = np.delete(a, 2)
            assert np.array_equal(a, want[name]), (r, name)


@pytest.mark.parametrize('n_ranks', [1, 3])
@pytest.mark.parametrize('by', BY)
def test_prescribed_owners_never_initialize_petsc(by, n_ranks):
    got = _launch(['partition_arrays_mpi.py', 'quad', by, 'brick'], n_ranks, blocked=True)
    assert got['petsc'] == '0', got
    assert got['digest'] == REFERENCE['quad %s brick %d' % (by, n_ranks)], got


def test_the_graph_partitioner_still_initializes_petsc():
    # the control for the test above: the probe does see PETSc when it is used
    got = _launch(['partition_arrays_mpi.py', 'quad', 'node', 'graph'], 3, blocked=True)
    assert got['petsc'] == '1', got


MG_CASES = {
    'quad': ['cube_mg_mpi.py', 'quad', 3, 3, 'mgm'],
    'hex': ['cube_mg_mpi.py', 'hex', 3, 3, 'mgm'],
    'p2': ['space_mg_mpi.py', 'p2', 3, 'mgm'],
    'ncp1': ['space_mg_mpi.py', 'ncp1', 3, 'mgm'],
    'dgp1': ['space_mg_mpi.py', 'dgp1', 3, 'mgm'],
}


# dgp1 partitions by elements, and its coarsest mesh has two cells a side, so at
# three ranks one brick is empty; the partitioner copes (test_a_rank_that_owns_nothing),
# but the DG quadrature set-up does not yet
@pytest.mark.parametrize('case,n_ranks', [(c, n) for c in ('quad', 'hex', 'p2', 'ncp1') for n in (1, 3, 4)] +
                         [('dgp1', 1), ('dgp1', 2)])
def test_multigrid_on_cartesian_bricks_without_petsc(case, n_ranks):
    pytest.importorskip('petsc4py')
    reference = _launch(MG_CASES[case], n_ranks)
    got = _launch(['with_cartesian_owners.py'] + MG_CASES[case], n_ranks, blocked=True)
    assert got['petsc_initialized'] == '0', got
    a, b = float(reference['checksum']), float(got['checksum'])
    assert abs(a - b) <= 1.0e-11 * abs(a), (reference, got)


def _grid_owners(points, h, nn, grid):
    """The brick owner of every node, worked out from its coordinates.

    The rule, stated independently of MeshTools._cartesianBrickStarts: the nn-1 cells along an
    axis are split into near-equal chunks (the partitioners' default partition), a node belongs
    to the chunk of the cell it starts -- the last node to the last cell's -- and an axis with
    fewer cells than ranks splits its nodes into chunks instead.
    """
    from proteus import MeshTools
    index = [np.rint(points[:, d] / h[d]).astype(int) if nn[d] > 1 else np.zeros(len(points), int)
             for d in range(3)]
    def chunkOf(d):
        n, p = nn[d], grid[d]
        if n > 1 and n - 1 >= p:
            return np.searchsorted(MeshTools._cartesianChunkStarts(n - 1, p)[1:],
                                   np.minimum(index[d], n - 2), side='right')
        return np.searchsorted(MeshTools._cartesianChunkStarts(n, p)[1:], index[d], side='right')
    chunk = [chunkOf(d) for d in range(3)]
    return chunk[0] + grid[0] * (chunk[1] + grid[1] * chunk[2])


@pytest.mark.parametrize('kind,nn,grid', [('tri', (7, 5, 1), (3, 2, 1)),
                                          ('quad', (7, 5, 1), (3, 2, 1)),
                                          ('quad', (4, 9, 1), (1, 4, 1)),
                                          ('tet', (5, 4, 3), (2, 2, 2)),
                                          ('hex', (5, 4, 3), (2, 1, 3)),
                                          ('hex', (3, 3, 3), (5, 1, 1))])
def test_cartesian_owners_follow_each_generators_numbering(kind, nn, grid):
    from proteus import MeshTools
    L = (1.0, 0.5, 2.0)
    mesh = {'tri': MeshTools.TriangularMesh, 'quad': MeshTools.QuadrilateralMesh,
            'tet': MeshTools.TetrahedralMesh, 'hex': MeshTools.HexahedralMesh}[kind]()
    if kind == 'tri':
        mesh.generateTriangularMeshFromRectangularGrid(nn[0], nn[1], L[0], L[1])
    elif kind == 'quad':
        mesh.generateQuadrilateralMeshFromRectangularGrid(nn[0], nn[1], L[0], L[1])
    elif kind == 'tet':
        mesh.generateTetrahedralMeshFromRectangularGrid(nn[0], nn[1], nn[2], *L)
    else:
        mesh.generateHexahedralMeshFromRectangularGrid(nn[0], nn[1], nn[2], *L)
    h = [L[d] / (nn[d] - 1) if nn[d] > 1 else 1.0 for d in range(3)]
    owners = MeshTools.cartesianNodeOwners(*nn, rankGrid=grid, fastest='y' if kind == 'quad' else 'x')
    assert owners.dtype == np.intc and owners.shape == (mesh.nNodes_global,)
    assert np.array_equal(owners, _grid_owners(np.asarray(mesh.nodeArray), h, nn, grid))
    # bricks are the default partition's near-equal chunks of the coarse cells (of the
    # nodes where there are fewer cells than ranks), so a 5-rank cut of 3 nodes leaves
    # two ranks empty instead of failing
    counts = np.bincount(owners, minlength=int(np.prod(grid)))
    assert counts.sum() == mesh.nNodes_global


def test_rank_grid_has_the_least_ghost_surface():
    from proteus import MeshTools
    assert MeshTools.cartesianRankGrid(64, 64, 1, 4) == (2, 2, 1)
    assert MeshTools.cartesianRankGrid(100, 10, 1, 4) == (4, 1, 1)
    assert MeshTools.cartesianRankGrid(10, 100, 1, 4) == (1, 4, 1)
    assert MeshTools.cartesianRankGrid(9, 9, 9, 8) == (2, 2, 2)
    assert MeshTools.cartesianRankGrid(5, 3, 1, 7) == (7, 1, 1)
    assert MeshTools.cartesianRankGrid(1, 1, 1, 1) == (1, 1, 1)
    with pytest.raises(ValueError):
        MeshTools.cartesianRankGrid(1, 1, 1, 2)
    for nn in [(17, 5, 1), (8, 8, 8), (30, 7, 3), (2, 50, 9)]:
        for size in range(1, 25):
            got = MeshTools.cartesianRankGrid(*nn, size)
            assert int(np.prod(got)) == size
            surface = lambda g: ((g[0] - 1) * nn[1] * nn[2] + (g[1] - 1) * nn[0] * nn[2] +
                                 (g[2] - 1) * nn[0] * nn[1])
            for px in range(1, size + 1):
                for py in range(1, size + 1):
                    if size % (px * py) == 0:
                        g = (px, py, size // (px * py))
                        if nn[2] == 1 and g[2] > 1:
                            continue
                        assert surface(got) <= surface(g), (nn, size, got, g)
