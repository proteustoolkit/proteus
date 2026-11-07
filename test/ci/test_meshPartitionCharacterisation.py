#!/usr/bin/env python
"""What the partitioner produces today, pinned before it is split (C32).

``proteus/partitioning.cpp`` is about to be cut into local, MPI and I/O parts.
These tests are written first, on the unchanged code, so that every step of
that refactor can show it changed nothing. Two kinds of check:

* **digests** (meshPartition_reference.json): every integer array each rank
  ends up with -- the eight offset and subdomain-to-global arrays, the
  old-to-new permutations, the subdomain mesh's connectivity and flags, and the
  quadratic and DG numberings built on top -- hashed over all ranks. A refactor
  that is bit-identical keeps every digest.
* **properties** against the serial mesh, which say what the arrays mean and
  so survive a deliberate change of numbering: the owned pieces reassemble the
  serial mesh, every entity once; a global number means the same entity on
  every rank that holds it; the ownership rules of partitionNodes; the overlap;
  the rank-interface flag -1 C0's kernels depend on.

The meshes are stored (meshPartition/), not generated: Triangle and TetGen
output differs between versions and platforms, and the digests would with it.
pslg (154 nodes) and plc (131 nodes) each have two regions (element flags 1
and 2) separated by an interior wall (face flag 5), tagged sides, and one side
deliberately untagged (flag 0), which a cut face must not be confused with.
TetGen's -1 markers on Steiner points and segment edges were replaced by 0.

Paths: ``node`` and ``element`` partition the mesh read into memory
(partitionNodes, partitionElements); ``files`` is ``parun -F``
(partitionNodesFromTriangleFiles / partitionNodesFromTetgenFiles), where the
partitioner reads the files itself. ``scatter`` prescribes owners; ``graph``
runs MatPartitioning with ``-mat_partitioning_type current``, the default
contiguous chunks, which is the only reproducible graph partition. Each case
checks it ran the entry point it is named for (meshPartition_mpi.py ``calls``).

Regenerate the reference only for an intended change, and say why in the
commit: ``PROTEUS_RECORD_PARTITION_REFERENCE=1 pytest test_meshPartitionCharacterisation.py``.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(HERE, 'meshPartition_reference.json')
RECORD = os.environ.get('PROTEUS_RECORD_PARTITION_REFERENCE') == '1'
with open(REFERENCE_FILE) as f:
    REFERENCE = json.load(f)

RANKS = [1, 2, 4]
#mesh, path, owners, overlap
CASES = [(m,) + c for m in ('tri', 'tet') for c in
         [('node', 'scatter', 1), ('node', 'graph', 1), ('node', 'scatter', 2),
          ('element', 'scatter', 1), ('element', 'graph', 1), ('element', 'scatter', 2),
          ('files', 'graph', 1)]]
ENTRY = {'node': 'partitionNodes', 'element': 'partitionElements',
         'files': {'tri': 'partitionNodesFromTriangleFiles', 'tet': 'partitionNodesFromTetgenFiles'}}
RANK_INTERFACE = -1

pytestmark = pytest.mark.xdist_group('meshPartition')


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(args, n_ranks, cwd, script='meshPartition_mpi.py', extraEnv=None):
    """Run script (meshPartition_mpi.py) on n_ranks (None: plain python, no launcher)."""
    env = dict(os.environ)
    env.update(extraEnv or {})
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    env['PETSC_OPTIONS'] = (env.get('PETSC_OPTIONS', '') + ' -mat_partitioning_type current').strip()
    cmd = [sys.executable, os.path.join(HERE, script)] + [str(a) for a in args]
    if n_ranks is not None:
        mpiexec = _mpiexec()
        if mpiexec is None:
            pytest.skip('no MPI launcher available')
        cmd = [mpiexec, '-n', str(n_ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    got = {}
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            got.update(item.split('=', 1) for item in line.split()[1:])
    assert got, result.stdout + result.stderr
    return got


_runs = {}


def run_files(d, mesh, n, filebase=None, extraEnv=None):
    """The -F path on n ranks for the stored mesh or a variant; (RESULT, ranks)."""
    os.makedirs(str(d), exist_ok=True)
    args = [mesh, 'files', 'graph', 1, os.path.join(str(d), 'out')] + ([filebase] if filebase else [])
    got = _launch(args, n, cwd=str(d), extraEnv=extraEnv)
    return got, [dict(np.load(os.path.join(str(d), 'out', 'rank%d.npz' % r))) for r in range(n)]


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """run(mesh, path, owners, overlap, n) -> (RESULT fields, [rank arrays]), each case run once."""
    def get(*key):
        if key not in _runs:
            mesh, path, owners, overlap, n = key
            d = tmp_path_factory.mktemp('%s-%s-%s-%d-%s' % key)
            got = _launch([mesh, path, owners, overlap, d / 'out'], n, cwd=d)
            ranks = [dict(np.load(d / 'out' / ('rank%d.npz' % r))) for r in range(n or 1)]
            _runs[key] = (got, ranks)
        return _runs[key]
    return get


@pytest.fixture(scope='module')
def serial(run):
    return lambda mesh: Serial(run(mesh, 'serial', 'scatter', 1, None)[1][0])


def digest(ranks):
    """sha256 over the integer arrays of every rank, in rank order."""
    total = hashlib.sha256()
    for arrays in ranks:
        h = hashlib.sha256()
        for name in sorted(arrays):
            a = arrays[name]
            if a.dtype.kind not in 'iu':
                continue
            h.update(name.encode())
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a.astype(np.int64)).tobytes())
        total.update(h.hexdigest().encode())
    return total.hexdigest()


class Serial(object):
    """The serial mesh, with every entity keyed by its sorted node numbers."""

    def __init__(self, g):
        self.nodeArray = g['global.nodeArray']
        self.key = {tuple(np.round(x, 12)): i for i, x in enumerate(self.nodeArray)}
        assert len(self.key) == len(self.nodeArray)
        self.nodeFlags = g['global.nodeMaterialTypes']
        self.elements = {tuple(sorted(e)): i for i, e in enumerate(g['global.elementNodesArray'])}
        self.elementFlags = g['global.elementMaterialTypes']
        self.faces = {tuple(sorted(f)): i for i, f in enumerate(g['global.elementBoundaryNodesArray'])}
        self.faceFlags = g['global.elementBoundaryMaterialTypes']
        self.interior = np.zeros(len(self.faces), bool)
        self.interior[g['global.interiorElementBoundariesArray']] = True
        self.edges = {tuple(sorted(e)): i for i, e in enumerate(g['global.edgeNodesArray'])}
        self.nodeStar = [set() for _ in self.nodeArray]
        for e in g['global.elementNodesArray']:
            for a in e:
                self.nodeStar[a].update(e)

    def nodesOf(self, arrays):
        """The serial number of each of a rank's subdomain nodes, by coordinates."""
        return np.array([self.key[tuple(np.round(x, 12))] for x in arrays['sub.nodeArray']], dtype=int)


class Piece(object):
    """One rank's subdomain, with its entities in serial numbering."""

    def __init__(self, serial, arrays, rank, size):
        a = arrays
        self.rank = rank
        node = serial.nodesOf(a)
        self.node = node
        self.element = np.array([serial.elements[tuple(sorted(node[e]))] for e in a['sub.elementNodesArray']], dtype=int)
        self.face = np.array([serial.faces[tuple(sorted(node[f]))] for f in a['sub.elementBoundaryNodesArray']], dtype=int)
        self.edge = np.array([serial.edges[tuple(sorted(node[e]))] for e in a['sub.edgeNodesArray']], dtype=int)
        self.serialOf = {'node': self.node, 'element': self.element, 'elementBoundary': self.face, 'edge': self.edge}
        self.offsets = {k: a['map.%sOffsets_subdomain_owned' % k] for k in self.serialOf}
        self.global_ = {k: a['map.%sNumbering_subdomain2global' % k] for k in self.serialOf}
        self.nOwned = {k: int(self.offsets[k][rank + 1] - self.offsets[k][rank]) for k in self.serialOf}
        for k in self.serialOf:
            assert len(self.offsets[k]) == size + 1
            assert len(self.global_[k]) == len(self.serialOf[k]), k


def pieces(serial, ranks):
    return [Piece(serial, a, r, len(ranks)) for r, a in enumerate(ranks)]


def nEntities(serial, kind):
    return {'node': len(serial.nodeArray), 'element': len(serial.elements),
            'elementBoundary': len(serial.faces), 'edge': len(serial.edges)}[kind]


@pytest.mark.parametrize('n', RANKS)
@pytest.mark.parametrize('case', CASES, ids=['-'.join(map(str, c)) for c in CASES])
def test_digest_is_pinned(run, case, n):
    mesh, path, owners, overlap = case
    got, ranks = run(*(case + (n,)))
    entry = ENTRY[path] if path != 'files' else ENTRY[path][mesh]
    #a serial node partition is given every node's owner (0) instead of running the graph partitioner
    given = int(owners == 'scatter' or (path in ('node', 'element') and n == 1))
    assert got['calls'] == '%s:%d' % (entry, given), got
    key = '%s %s %s %d %d' % (mesh, path, owners, overlap, n)
    d = digest(ranks)
    if RECORD:
        REFERENCE[key] = d
        with open(REFERENCE_FILE, 'w') as f:
            json.dump(REFERENCE, f, indent=1, sort_keys=True)
            f.write('\n')
    assert d == REFERENCE.get(key), key


@pytest.mark.parametrize('n', RANKS)
@pytest.mark.parametrize('case', CASES, ids=['-'.join(map(str, c)) for c in CASES])
def test_owned_pieces_reassemble_the_serial_mesh(run, serial, case, n):
    """Every node, element, face and edge is owned by exactly one rank; the
    global numbers of each kind are 0..N-1, the owned ones of rank r the
    contiguous range offsets[r]:offsets[r+1] at the front of its subdomain; and
    the flags are the serial mesh's (a face may also read -1, see below)."""
    S = serial(case[0])
    P = pieces(S, run(*(case + (n,)))[1])
    for kind in ('node', 'element', 'elementBoundary', 'edge'):
        N = nEntities(S, kind)
        offsets = P[0].offsets[kind]
        assert offsets[0] == 0 and offsets[-1] == N, (kind, offsets)
        assert np.all(np.diff(offsets) >= 0), (kind, offsets)
        owned = np.concatenate([p.serialOf[kind][:p.nOwned[kind]] for p in P])
        assert np.array_equal(np.sort(owned), np.arange(N)), kind
        for p in P:
            assert np.array_equal(p.offsets[kind], offsets), kind
            assert np.array_equal(p.global_[kind][:p.nOwned[kind]],
                                  np.arange(offsets[p.rank], offsets[p.rank + 1])), (kind, p.rank)
    for p, arrays in zip(P, run(*(case + (n,)))[1]):
        assert np.array_equal(arrays['sub.nodeMaterialTypes'], S.nodeFlags[p.node])
        assert np.array_equal(arrays['sub.elementMaterialTypes'], S.elementFlags[p.element])
        flags = arrays['sub.elementBoundaryMaterialTypes']
        same = flags == S.faceFlags[p.face]
        assert np.all(same | (flags == RANK_INTERFACE)), p.rank


@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('case', CASES, ids=['-'.join(map(str, c)) for c in CASES])
def test_a_global_number_means_the_same_entity_on_every_rank(run, serial, case, n):
    """The subdomain-to-global maps of all ranks agree: a ghost carries the
    number its owner gave it. Ghosts follow the owned entities in increasing
    global order and lie outside the rank's own range."""
    S = serial(case[0])
    P = pieces(S, run(*(case + (n,)))[1])
    for kind in ('node', 'element', 'elementBoundary', 'edge'):
        meaning = {}
        for p in P:
            for g, s in zip(p.global_[kind], p.serialOf[kind]):
                assert meaning.setdefault(int(g), int(s)) == s, (kind, p.rank, g)
            ghosts = p.global_[kind][p.nOwned[kind]:]
            lo, hi = p.offsets[kind][p.rank], p.offsets[kind][p.rank + 1]
            assert np.all((ghosts < lo) | (ghosts >= hi)), (kind, p.rank)
            if case[1] != 'files':
                assert np.all(np.diff(ghosts) > 0), (kind, p.rank)
        assert len(set(meaning.values())) == len(meaning), kind


NODE_CASES = [c for c in CASES if c[1] == 'node']


@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('case', NODE_CASES, ids=['-'.join(map(str, c)) for c in NODE_CASES])
def test_partitionNodes_ownership_rules(run, serial, case, n):
    """Given the node owners, partitionNodes decides the rest locally: an
    element and a face belong to the lowest rank that owns one of their nodes,
    an edge to the owner of its end with the lower new number. Owned entities
    keep their old (serial) order within a rank. These are the rules the
    local part split out of partitionNodes must reproduce."""
    S = serial(case[0])
    P = pieces(S, run(*(case + (n,)))[1])
    newNumber = {}
    owner = {}
    for kind in ('node', 'element', 'elementBoundary', 'edge'):
        newNumber[kind] = np.empty(nEntities(S, kind), int)
        owner[kind] = np.empty(nEntities(S, kind), int)
        for p in P:
            newNumber[kind][p.serialOf[kind]] = p.global_[kind]
            owner[kind][p.serialOf[kind][:p.nOwned[kind]]] = p.rank
            assert np.all(np.diff(p.serialOf[kind][:p.nOwned[kind]]) > 0), (kind, p.rank)
    nodeOwner = owner['node']
    for entities, kind in ((S.elements, 'element'), (S.faces, 'elementBoundary')):
        for nodes, i in entities.items():
            assert owner[kind][i] == min(nodeOwner[a] for a in nodes), (kind, i)
    for (a, b), i in S.edges.items():
        assert owner['edge'][i] == nodeOwner[a if newNumber['node'][a] < newNumber['node'][b] else b], i


@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('case', NODE_CASES, ids=['-'.join(map(str, c)) for c in NODE_CASES])
def test_partitionNodes_overlap(run, serial, case, n):
    """Overlap 1: a rank's elements are those with an owned node. Each further
    layer adds the elements of the nodes the last layer reached through a node
    star, the overlap nodes being the nodes of the subdomain's elements."""
    S = serial(case[0])
    P = pieces(S, run(*(case + (n,)))[1])
    elementsOf = [set() for _ in S.nodeArray]
    for nodes, i in S.elements.items():
        for a in nodes:
            elementsOf[a].add(i)
    for p in P:
        owned = set(p.node[:p.nOwned['node']].tolist())
        elements = set().union(*(elementsOf[a] for a in owned)) if owned else set()
        reached = set(owned)
        last = set(owned)
        for layer in range(case[3] - 1):
            last = set().union(*(S.nodeStar[a] for a in last)) - reached
            reached |= last
            elements |= set().union(*(elementsOf[a] for a in last)) if last else set()
        assert set(p.element.tolist()) == elements, p.rank
        nodes = set(owned)
        for e, i in S.elements.items():
            if i in elements:
                nodes.update(e)
        assert set(p.node.tolist()) == nodes, p.rank


@pytest.mark.parametrize('n', [2, 4])
@pytest.mark.parametrize('case', CASES, ids=['-'.join(map(str, c)) for c in CASES])
def test_rank_interface_flag(run, serial, case, n):
    """A face the partition cut -- exterior to the subdomain, interior to the
    mesh -- reads -1 where the mesh gave it no flag (0), and keeps a positive
    flag (the interior wall, 5). A real boundary face keeps its flag, 0
    included. The ``files`` path (parun -F) has no global mesh to compare
    with; its partitioners ask each face's owner (markRankInterfacesByOwner)
    and mark the same faces."""
    S = serial(case[0])
    got, ranks = run(*(case + (n,)))
    P = pieces(S, ranks)
    nCut = nWall = 0
    for p, a in zip(P, ranks):
        flags = a['sub.elementBoundaryMaterialTypes']
        exterior = a['sub.exteriorElementBoundariesArray']
        serialFace = p.face[exterior]
        cut = S.interior[serialFace]
        want = S.faceFlags[serialFace].copy()
        want[cut & (want == 0)] = RANK_INTERFACE
        assert np.array_equal(flags[exterior], want), p.rank
        interior = np.setdiff1d(np.arange(len(flags)), exterior)
        assert np.all(flags[interior] != RANK_INTERFACE), p.rank
        nCut += int(np.sum(cut))
        nWall += int(np.sum(cut & (S.faceFlags[serialFace] == 5)))
    _cuts[case + (n,)] = (nCut, nWall)


_cuts = {}


@pytest.mark.parametrize('case,wall', [(('tri', 'node', 'scatter', 1, 4), True), (('tri', 'node', 'graph', 1, 2), False),
                                       (('tri', 'element', 'scatter', 1, 4), True), (('tri', 'files', 'graph', 1, 2), False),
                                       (('tet', 'node', 'scatter', 1, 2), True), (('tet', 'node', 'graph', 1, 4), True),
                                       (('tet', 'element', 'scatter', 1, 2), True), (('tet', 'files', 'graph', 1, 4), True)],
                         ids=lambda c: '-'.join(map(str, c)) if isinstance(c, tuple) else 'wall%d' % c)
def test_rank_interface_cases_cut_faces_and_the_wall(run, serial, case, wall):
    """test_rank_interface_flag means something only where the partition cuts
    faces, and its "keep a positive flag" branch only where it cuts the wall.
    With scattered owners and overlap 2 every subdomain is the whole mesh and
    nothing is cut, and the triangle mesh's contiguous chunks never cross the
    wall, so these are the cases that do (counts measured 2026-11-03)."""
    test_rank_interface_flag(run, serial, case[:4], case[4])
    nCut, nWall = _cuts[case]
    assert nCut > 0 and (nWall > 0) == wall, (nCut, nWall)


@pytest.mark.parametrize('n', RANKS)
@pytest.mark.parametrize('case', CASES, ids=['-'.join(map(str, c)) for c in CASES])
def test_quadratic_and_dg_numberings(run, serial, case, n):
    """The P2 numbering (nodes, then faces in 2D or edges in 3D) and the DG P1
    numbering (d+1 per element) built on the partition: their owned ranges
    cover 0..N-1 once, and a global P2 dof is the same Lagrange node on every
    rank."""
    mesh = case[0]
    S = serial(mesh)
    _, ranks = run(*(case + (n,)))
    nd = 2 if mesh == 'tri' else 3
    nP2 = len(S.nodeArray) + (len(S.faces) if nd == 2 else len(S.edges))
    point = {}
    for r, a in enumerate(ranks):
        nAll, nSub, _ = a['quadratic.counts']
        assert nAll == nP2
        offsets = a['quadratic.dof_offsets_subdomain_owned']
        assert offsets[0] == 0 and offsets[-1] == nP2
        g = a['quadratic.subdomain2global'][:nSub]
        assert np.array_equal(g[:offsets[r + 1] - offsets[r]], np.arange(offsets[r], offsets[r + 1]))
        for dof, x in zip(g, a['quadratic.lagrangeNodesArray'][:nSub]):
            key = tuple(np.round(x, 10))
            assert point.setdefault(int(dof), key) == key, (r, dof)
    assert len(point) == nP2
    if 'dg.counts' in ranks[0]:
        for r, a in enumerate(ranks):
            assert a['dg.counts'][0] == len(S.elements) * (nd + 1)
            offsets = a['dg.dof_offsets_subdomain_owned']
            assert np.array_equal(offsets, ranks[0]['dg.dof_offsets_subdomain_owned'])
            assert offsets[-1] == len(S.elements) * (nd + 1)


def _readMesh(mesh):
    from proteus import MeshTools
    filebase = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    if mesh == 'tri':
        m = MeshTools.TriangularMesh()
        m.generateFromTriangleFiles(filebase=filebase, base=1)
    else:
        m = MeshTools.TetrahedralMesh()
        m.generateFromTetgenFiles(filebase, 1)
    return m


def _nodeOwners(owners, nNodes, size):
    """The owners meshPartition_mpi.py's run used: its scatter hash, or the
    default contiguous chunks, which is what MatPartitioning 'current' keeps."""
    i = np.arange(nNodes, dtype=np.int64)
    if owners == 'scatter':
        return (((i * 2654435761) >> 7) % size).astype('i')
    starts = np.cumsum([0] + [nNodes // size + (nNodes % size > r) for r in range(size)])
    return (np.searchsorted(starts[1:], i, side='right')).astype('i')


@pytest.mark.parametrize('n', RANKS)
@pytest.mark.parametrize('case', NODE_CASES, ids=['-'.join(map(str, c)) for c in NODE_CASES])
def test_partitionNodesLocal_computes_every_rank_in_one_process(run, case, n):
    """partitionNodes is its owners (the MPI part: prescribed, or gathered from
    the graph partitioner) followed by partitionNodesLocal, which needs no MPI.
    Called here in this one process for each rank of n in turn, with the owners
    the MPI run used, it gives that rank's arrays bit for bit: the maps, the
    permutations and the subdomain mesh. (The run's face flags were then marked
    -1 on cut faces by MeshTools, which partitionNodesLocal does not do.)"""
    from proteus import cmeshTools, cpartitioning
    mesh, path, owners, overlap = case
    _, ranks = run(*(case + (n,)))
    for r in range(n):
        m = _readMesh(mesh)
        sub = cmeshTools.CMesh()
        got = cpartitioning.partitionNodesLocal(r, n, overlap, m.cmesh, sub,
                                                _nodeOwners(owners, m.nNodes_global, n))
        want = ranks[r]
        for name, a in zip(MAPS_RETURNED, got):
            assert np.array_equal(a, want['map.' + name]), (r, name)
        assert np.array_equal(cpartitioning.getNodeNumbering_global_old2new(m.cmesh),
                              want['map.nodeNumbering_global_old2new']), r
        for name, a in zip(SUBDOMAIN_INTERFACE, cmeshTools.buildPythonMeshInterface(sub)):
            if a is None or np.asarray(a).dtype.kind not in 'iu':
                continue
            b = want['sub.' + name]
            if name == 'elementBoundaryMaterialTypes':
                b = np.where(b == RANK_INTERFACE, 0, b)
            assert np.array_equal(np.asarray(a), b), (r, name)


MAPS_RETURNED = ('elementOffsets_subdomain_owned', 'elementNumbering_subdomain2global',
                 'nodeOffsets_subdomain_owned', 'nodeNumbering_subdomain2global',
                 'elementBoundaryOffsets_subdomain_owned', 'elementBoundaryNumbering_subdomain2global',
                 'edgeOffsets_subdomain_owned', 'edgeNumbering_subdomain2global')
#the order of cmeshTools.buildPythonMeshInterface, as in meshPartition_mpi.py
SUBDOMAIN_INTERFACE = ('nElements_global nNodes_global nNodes_element nNodes_elementBoundary '
                       'nElementBoundaries_element nElementBoundaries_global '
                       'nInteriorElementBoundaries_global nExteriorElementBoundaries_global '
                       'max_nElements_node nEdges_global max_nNodeNeighbors_node elementNodesArray '
                       'nodeElementsArray nodeElementOffsets elementNeighborsArray elementBoundariesArray '
                       'elementBoundaryNodesArray elementBoundaryElementsArray '
                       'elementBoundaryLocalElementBoundariesArray interiorElementBoundariesArray '
                       'exteriorElementBoundariesArray edgeNodesArray nodeStarArray nodeStarOffsets '
                       'elementMaterialTypes elementBoundaryMaterialTypes nodeMaterialTypes').split()


def _regularMesh(kind):
    from proteus import MeshTools
    if kind == 'quad':
        m = MeshTools.QuadrilateralMesh()
        m.generateQuadrilateralMeshFromRectangularGrid(7, 5, 1.0, 0.7)
    else:
        m = MeshTools.HexahedralMesh()
        m.generateHexahedralMeshFromRectangularGrid(5, 4, 3, 1.0, 0.7, 1.3)
    return m


@pytest.mark.parametrize('size', [1, 3, 4])
@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
def test_ownership_rules_in_process(kind, size):
    """elementOwnersFromNodeOwners, elementBoundaryOwnersFromNodeOwners and
    edgeOwnersFromNodeOwners, called directly: the rules
    test_partitionNodes_ownership_rules checks on MPI output, here for every
    element kind, including the quadrilateral and hexahedron branches of the
    element boundary rule that the stored simplex meshes do not reach."""
    from proteus import cpartitioning
    m = _readMesh(kind) if kind in ('tri', 'tet') else _regularMesh(kind)
    owners = _nodeOwners('scatter', m.nNodes_global, size)
    elementOwner, boundaryOwner, edgeOwner, old2new = cpartitioning.ownersFromNodeOwners(m.cmesh, owners, size)
    assert np.array_equal(elementOwner, owners[np.asarray(m.elementNodesArray)].min(axis=1))
    assert np.array_equal(boundaryOwner, owners[np.asarray(m.elementBoundaryNodesArray)].min(axis=1))
    edges = np.asarray(m.edgeNodesArray)
    lower = np.where(old2new[edges[:, 0]] < old2new[edges[:, 1]], edges[:, 0], edges[:, 1])
    assert np.array_equal(edgeOwner, owners[lower])
    if size > 1:
        assert len(set(boundaryOwner.tolist())) > 1


@pytest.mark.parametrize('size', [1, 2, 4])
@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
def test_graph_partitioner_input_in_process(kind, size):
    """What graphNodeOwners hands MatPartitioning on each rank: the rows of the
    node graph for its default chunk, each node's star sorted. With
    -mat_partitioning_type current the digests cannot see this input (the
    partition ignores it), so it is checked here against the mesh's node star."""
    from proteus import cpartitioning
    m = _readMesh(kind) if kind in ('tri', 'tet') else _regularMesh(kind)
    n = m.nNodes_global
    starts = np.cumsum([0] + [n // size + (n % size > r) for r in range(size)])
    starOffsets, star = np.asarray(m.nodeStarOffsets), np.asarray(m.nodeStarArray)
    for r in range(size):
        offsets, neighbors = cpartitioning.nodeGraphRows(m.cmesh, int(starts[r]), int(starts[r + 1]))
        rows = [np.sort(star[starOffsets[a]:starOffsets[a + 1]]) for a in range(starts[r], starts[r + 1])]
        assert np.array_equal(offsets, np.cumsum([0] + [len(x) for x in rows]))
        assert np.array_equal(neighbors, np.concatenate(rows) if rows else np.empty(0, int))


ELEMENT_CASES = [c for c in CASES if c[1] == 'element']


@pytest.mark.parametrize('n', RANKS)
@pytest.mark.parametrize('case', ELEMENT_CASES, ids=['-'.join(map(str, c)) for c in ELEMENT_CASES])
def test_partitionElementsLocal_computes_every_rank_in_one_process(run, case, n):
    """As test_partitionNodesLocal_computes_every_rank_in_one_process, for
    partitionElements: its owners (prescribed, or the graph partitioner's,
    gathered) followed by partitionElementsLocal, which needs no MPI and is
    called here for each rank in turn on a freshly read mesh (it renumbers the
    global mesh in place)."""
    from proteus import cmeshTools, cpartitioning
    mesh, path, owners, overlap = case
    _, ranks = run(*(case + (n,)))
    for r in range(n):
        m = _readMesh(mesh)
        sub = cmeshTools.CMesh()
        got = cpartitioning.partitionElementsLocal(r, n, overlap, m.cmesh, sub,
                                                   _nodeOwners(owners, m.nElements_global, n))
        want = ranks[r]
        for name, a in zip(MAPS_RETURNED, got):
            assert np.array_equal(a, want['map.' + name]), (r, name)
        assert np.array_equal(cpartitioning.getElementNumbering_global_old2new(m.cmesh),
                              want['map.elementNumbering_global_old2new']), r
        for name, a in zip(SUBDOMAIN_INTERFACE, cmeshTools.buildPythonMeshInterface(sub)):
            if a is None or np.asarray(a).dtype.kind not in 'iu':
                continue
            b = want['sub.' + name]
            if name == 'elementBoundaryMaterialTypes':
                b = np.where(b == RANK_INTERFACE, 0, b)
            assert np.array_equal(np.asarray(a), b), (r, name)


@pytest.mark.parametrize('size', [1, 2, 4])
@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
def test_element_graph_partitioner_input_in_process(kind, size):
    """What graphElementOwners hands MatPartitioning on each rank: its default
    chunk of the element graph, each element's neighbours across its element
    boundaries (none across the domain boundary), sorted."""
    from proteus import cpartitioning
    m = _readMesh(kind) if kind in ('tri', 'tet') else _regularMesh(kind)
    n = m.nElements_global
    starts = np.cumsum([0] + [n // size + (n % size > r) for r in range(size)])
    neighborsOf = np.asarray(m.elementNeighborsArray)
    for r in range(size):
        offsets, neighbors = cpartitioning.elementGraphRows(m.cmesh, int(starts[r]), int(starts[r + 1]))
        rows = [np.sort(neighborsOf[e][neighborsOf[e] >= 0]) for e in range(starts[r], starts[r + 1])]
        assert np.array_equal(offsets, np.cumsum([0] + [len(x) for x in rows]))
        assert np.array_equal(neighbors, np.concatenate(rows) if rows else np.empty(0, int))


@pytest.mark.parametrize('size', [1, 3, 4])
@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
def test_element_ownership_rules_in_process(kind, size):
    """nodeOwnersFromElementOwners, elementBoundaryOwnersFromElementOwners and
    edgeOwnersFromElementOwners: a node, a face and an edge belong to the lowest
    rank owning an element that contains them (all of the face's or edge's
    nodes). Stated here independently, from the element-node incidence."""
    from proteus import cpartitioning
    m = _readMesh(kind) if kind in ('tri', 'tet') else _regularMesh(kind)
    owners = _nodeOwners('scatter', m.nElements_global, size)
    nodeOwner, boundaryOwner, edgeOwner = cpartitioning.ownersFromElementOwners(m.cmesh, owners, size)
    elements = np.asarray(m.elementNodesArray)
    want = np.full(m.nNodes_global, size)
    np.minimum.at(want, elements.ravel(), np.repeat(owners, elements.shape[1]))
    assert np.array_equal(nodeOwner, want)
    elementsOf = [set() for _ in range(m.nNodes_global)]
    for e, nodes in enumerate(elements):
        for a in nodes:
            elementsOf[a].add(e)
    def lowest(nodes):
        common = set.intersection(*(elementsOf[a] for a in nodes))
        return min(owners[e] for e in common)
    assert np.array_equal(boundaryOwner, [lowest(f) for f in np.asarray(m.elementBoundaryNodesArray)])
    assert np.array_equal(edgeOwner, [lowest(e) for e in np.asarray(m.edgeNodesArray)])
    if size > 1:
        assert len(set(nodeOwner.tolist())) > 1


@pytest.mark.parametrize('size', [1, 2, 4])
@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_reader_node_stars(mesh, size):
    """The first read of the parun -F partitioners (Triangle and TetGen): each
    rank streams the .ele file and keeps the star of every node in its default
    chunk. That is the in-memory reader's node star for the same nodes."""
    from proteus import cpartitioning
    m = _readMesh(mesh)
    n = m.nNodes_global
    starts = np.cumsum([0] + [n // size + (n % size > r) for r in range(size)])
    starOffsets, star = np.asarray(m.nodeStarOffsets), np.asarray(m.nodeStarArray)
    filebase = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    simplexDim = {'tri': 3, 'tet': 4}[mesh]
    for r in range(size):
        nElements, stars = cpartitioning.readNodeStars(filebase, simplexDim, 1, n, int(starts[r]), int(starts[r + 1]))
        assert nElements == m.nElements_global
        assert len(stars) == starts[r + 1] - starts[r]
        for k, a in enumerate(range(starts[r], starts[r + 1])):
            assert np.array_equal(stars[k], np.sort(star[starOffsets[a]:starOffsets[a + 1]])), (r, a)
    with pytest.raises(IOError):
        cpartitioning.readNodeStars(filebase + '_missing', simplexDim, 1, n, 0, n)


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_reader_subdomain_vertices(mesh):
    """The last read of the parun -F partitioners (Triangle and TetGen): each
    rank streams the .node file and keeps its owned and overlap vertices in the
    new numbering. That is the in-memory reader's nodes, permuted. Nodes the
    subdomain touches without keeping them, and nodes it does not touch, are
    skipped; without markers the flags are left alone."""
    from proteus import cpartitioning
    m = _readMesh(mesh)
    n = m.nNodes_global
    nodeArray, flags = np.asarray(m.nodeArray), np.asarray(m.nodeMaterialTypes)
    filebase = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    nSpace = {'tri': 2, 'tet': 3}[mesh]
    rng = np.random.default_rng(32)
    new2old = rng.permutation(n)
    old2new = np.argsort(new2old)
    begin, end = n // 3, 2 * n // 3
    others = np.r_[0:begin, end:n]
    overlap = np.sort(rng.choice(others, size=n // 10, replace=False))
    touched = np.r_[begin:end, overlap, rng.choice(np.setdiff1d(others, overlap), size=5, replace=False)]
    nodes_old2new = {int(new2old[g]): int(g) for g in touched}
    for markers in (1, 0):
        s2g, g2s, xyz, mt = cpartitioning.readSubdomainVertices(filebase, nSpace, 1, markers, nodes_old2new,
                                                                begin, end, overlap.tolist())
        kept = np.r_[begin:end, overlap]
        assert np.array_equal(s2g, kept)
        assert g2s == {int(g): k for k, g in enumerate(kept)}
        assert np.array_equal(xyz, nodeArray[new2old[kept]])
        assert np.array_equal(mt, flags[new2old[kept]] if markers else np.zeros(len(kept)))
    assert flags.any()
    assert np.array_equal(old2new[new2old], np.arange(n))
    with pytest.raises(IOError):
        cpartitioning.readSubdomainVertices(filebase + '_missing', nSpace, 1, 1, nodes_old2new, begin, end, [])


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_reader_subdomain_elements(mesh):
    """The second read of the parun -F partitioners (Triangle and TetGen): each
    rank streams the .ele file again and keeps, in file order, the elements
    with a node it owns after repartitioning, with their file nodes and first
    attribute. Checked against the file itself and the in-memory reader."""
    from proteus import cpartitioning
    m = _readMesh(mesh)
    n = m.nNodes_global
    filebase = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    simplexDim = {'tri': 3, 'tet': 4}[mesh]
    table = np.loadtxt(filebase + '.ele', skiprows=1, comments='#', ndmin=2)
    fileElements = table[:, 0].astype(int) - 1
    fileNodes = table[:, 1:1 + simplexDim].astype(int) - 1
    fileIds = table[:, 1 + simplexDim]
    assert np.array_equal(np.sort(fileNodes, axis=1), np.sort(np.asarray(m.elementNodesArray), axis=1))
    assert np.array_equal(fileIds, np.asarray(m.elementMaterialTypes))
    rng = np.random.default_rng(32)
    for nodes in (rng.choice(n, size=n // 4, replace=False), np.arange(n), np.array([], dtype=int)):
        nElements, markers, elements, elementNodes, elementIds = cpartitioning.readSimplexElementsTouching(
            filebase, simplexDim, 1, n, nodes.tolist())
        keep = np.isin(fileNodes, nodes).any(axis=1)
        assert nElements == m.nElements_global and markers == 1
        assert np.array_equal(elements, fileElements[keep])
        assert np.array_equal(elementNodes, fileNodes[keep])
        assert np.array_equal(elementIds, fileIds[keep])
    with pytest.raises(IOError):
        cpartitioning.readSimplexElementsTouching(filebase + '_missing', simplexDim, 1, n, [0])


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_files_node_number_exchange(tmp_path, n):
    """The parun -F partitioners number the nodes of their second element read
    by exchange, not by reading mappings.h5: each rank sends old->new of its
    default chunk to the new owner, and asks the chunk owners for the new
    numbers of the other nodes its elements touch. Checked under a random
    permutation and uneven new ranges (meshPartition_exchange_mpi.py), which the
    stored-mesh cases never produce. The element, face and edge numberings go
    through a directory on the default chunks (subdomainNumbers), checked in
    the same script under random owners and random ghosts."""
    for nNodes, seed in ((1, 0), (37, 1), (1000, 2)):
        got = _launch([nNodes, seed], n, cwd=tmp_path, script='meshPartition_exchange_mpi.py')
        assert got == {'ranks': str(n), 'owned': str(nNodes), 'asked': got['asked'],
                       'directory': got['directory']}


@pytest.mark.parametrize('n', [1, 2, 3, 4])
@pytest.mark.parametrize('case,message', [
    ('directory', 'entity 0 (asked by rank {asker}, of rank 0) has no owner'),
    ('range', 'entity 37 (asked by rank {asker}) is outside [0,37)'),
    ('node', 'node 37 (asked by rank {asker}) is outside [0,37)')])
def test_files_lookup_of_a_number_no_rank_holds_ends_the_run(tmp_path, case, message, n):
    """The -F partitioners' directory lookups (askOwningRanks) asked for a
    number no rank can answer: an entity of a subdomain that no rank owns came
    back as -1 and was used as a new number, and a number outside the ranks'
    ranges was an out-of-bounds access (SEGV); the asserts that would have
    caught them are compiled out. Each now ends the run on every rank with a
    message naming the number and the rank that asked
    (meshPartition_missing_owner_mpi.py)."""
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    cmd = [mpiexec, '-n', str(n), sys.executable, os.path.join(HERE, 'meshPartition_missing_owner_mpi.py'), case, '37']
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert 'RESULT' not in result.stdout, out
    assert message.format(asker=n - 1) in out, out


@pytest.mark.parametrize('ext,nNodes,headerHasMarkers', [('pslg.edge', 2, True), ('pslg.edge', 2, False),
                                                         ('plc.face', 3, True), ('plc.edge', 2, False)])
def test_files_reader_faces_and_edges(ext, nNodes, headerHasMarkers):
    """The .face/.edge reads of the parun -F partitioners: each rank streams
    the file and keeps, in file order, the entities with a node of one of its
    subdomain's elements; the partitioners do nothing with the others. Both
    read Triangle's .edge as element boundaries with markers, and every .edge
    as edges without (the header's marker count and the marker column are not
    read: what they have always done)."""
    from proteus import cpartitioning
    filename = os.path.join(HERE, 'meshPartition', ext)
    table = np.loadtxt(filename, skiprows=1, comments='#', ndmin=2).astype(int)
    numbers, nodes, markers = table[:, 0] - 1, table[:, 1:1 + nNodes] - 1, table[:, 1 + nNodes]
    n = nodes.max() + 1
    rng = np.random.default_rng(32)
    for keep in (rng.choice(n, size=n // 5, replace=False), np.arange(n), np.array([], dtype=int)):
        count, hasMarkers, got, gotNodes, gotMarkers = cpartitioning.readSimplexSubentitiesTouching(
            filename, nNodes, 1, headerHasMarkers, keep.tolist())
        touches = np.isin(nodes, keep).any(axis=1)
        assert count == len(table) and hasMarkers == int(headerHasMarkers)
        assert np.array_equal(got, numbers[touches])
        assert np.array_equal(gotNodes, nodes[touches])
        assert np.array_equal(gotMarkers, markers[touches] if headerHasMarkers else [])
    if headerHasMarkers:
        assert markers.any()
    with pytest.raises(IOError):
        cpartitioning.readSimplexSubentitiesTouching(filename + '_missing', nNodes, 1, headerHasMarkers, [0])


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_path_refuses_element_partitioning(tmp_path, mesh):
    """parun -F partitions by nodes only. Asking it for an element partition
    used to stop at a bare assert in generatePartitionedMeshFrom*Files, past
    which partitionMeshFromFiles called cpartitioning.partitionElementsFromTetgenFiles,
    which does not exist. Now every rank raises a ValueError that says what to
    do, before any partitioner runs. 2 ranks."""
    got = _launch([mesh, 'files-element', 'graph', 1, tmp_path / 'out'], 2, cwd=tmp_path)
    assert got == {'mesh': mesh, 'path': 'files-element', 'ranks': '2', 'errors': 'ValueError,ValueError',
                   'calls': 'none'}


def _variantMesh(tmp_path, mesh, vertexFlags):
    """The stored mesh with its vertex flags as a TetGen/Triangle attribute
    (header ``n d 1 0``, same column) or dropped (``n d 0 0``); the other files
    copied. Returns the filebase."""
    stored = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    base = str(tmp_path / ('%s_%s' % (os.path.basename(stored), vertexFlags)))
    for ext in ('ele', 'edge', 'face') if mesh == 'tet' else ('ele', 'edge'):
        shutil.copy(stored + '.' + ext, base + '.' + ext)
    with open(stored + '.node') as f:
        lines = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    n, d = lines[0].split()[:2]
    assert lines[0].split()[2:] == ['0', '1']
    rows = [l.split() for l in lines[1:]]
    if vertexFlags == 'attribute':
        out = ['%s %s 1 0' % (n, d)] + [' '.join(r) for r in rows]
    else:
        out = ['%s %s 0 0' % (n, d)] + [' '.join(r[:1 + int(d)]) for r in rows]
    with open(base + '.node', 'w') as f:
        f.write('\n'.join(out) + '\n')
    return base


@pytest.mark.parametrize('n', [2, 4])
def test_files_tetgen_vertex_attribute_is_the_node_flag_on_every_rank(tmp_path, n):
    """A TetGen .node with an attribute and no marker: the -F partitioner
    takes the attribute as the node flag (and warns), as the in-memory reader
    does. It used to do so on rank 0 only (the rank==0 guard covered the
    switch, not just the warning), so the other ranks' node flags came from the
    faces instead. With the stored mesh's markers moved to an attribute
    column, every array on every rank equals the stored mesh's."""
    base = _variantMesh(tmp_path, 'tet', 'attribute')
    _, stored = run_files(tmp_path / 'stored', 'tet', n)
    _, attr = run_files(tmp_path / 'attr', 'tet', n, base)
    assert len(attr) == len(stored) == n
    for a, b in zip(attr, stored):
        assert sorted(a) == sorted(b)
        for name in a:
            assert np.array_equal(a[name], b[name]), name


#fill fresh heap memory with a pattern (macOS: 0xAA, glibc: ~0x55 = 0xAA), so a
#read of memory nobody wrote shows up as -1431655766 instead of a lucky 0
SCRIBBLE = {'MallocScribble': '1', 'MALLOC_PERTURB_': '85'}


@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_node_flags_without_vertex_markers_are_initialised(tmp_path, mesh):
    """A .node file without vertex markers: the -F partitioners give the nodes
    of exterior subdomain faces the face flags and every other node 0. The
    others used to keep what the heap held (new int[] never written); with the
    allocator scribbling, they read -1431655766. 2 ranks, Triangle and TetGen:
    the arrays do not depend on the heap, every flag is 0 or a face flag, and a
    node on no exterior face reads 0."""
    base = _variantMesh(tmp_path, mesh, 'none')
    _, plain = run_files(tmp_path / 'plain', mesh, 2, base)
    _, scribbled = run_files(tmp_path / 'scribbled', mesh, 2, base, extraEnv=SCRIBBLE)
    for a, b in zip(plain, scribbled):
        for name in a:
            assert np.array_equal(a[name], b[name]), name
        flags = a['sub.nodeMaterialTypes']
        faceFlags = a['sub.elementBoundaryMaterialTypes']
        faceNodes = a['sub.elementBoundaryNodesArray']
        exterior = a['sub.exteriorElementBoundariesArray']
        onExterior = np.zeros(len(flags), bool)
        onExterior[faceNodes[exterior].ravel()] = True
        assert np.all(flags[~onExterior] == 0)
        assert set(flags[onExterior].tolist()) <= set(np.maximum(faceFlags, 0).tolist())
        assert onExterior.any() and (~onExterior).any()



@pytest.mark.parametrize('mesh', ['tri', 'tet'])
def test_files_marks_rank_interfaces_without_face_flags(tmp_path, mesh):
    """Element-boundary files without markers (.face, or Triangle's .edge, header
    ``n 0``): the subdomain builder flags every subdomain-exterior face 1, cut
    faces included, so marking only flag-0 faces marked nothing. The -F path
    must mark the same faces -1 as the in-memory node partition of the same
    files on the same ranks: the cut faces, here all of them. 2 ranks."""
    stored = os.path.join(HERE, 'meshPartition', {'tri': 'pslg', 'tet': 'plc'}[mesh])
    boundary = 'face' if mesh == 'tet' else 'edge'
    base = str(tmp_path / 'nofaceflags')
    for ext in ('node', 'ele', 'edge', 'face') if mesh == 'tet' else ('node', 'ele', 'edge'):
        shutil.copy(stored + '.' + ext, base + '.' + ext)
    with open(stored + '.' + boundary) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    nNodes = 3 if boundary == 'face' else 2
    with open(base + '.' + boundary, 'w') as f:
        f.write('\n'.join(['%s 0' % lines[0][0]] + [' '.join(l[:1 + nNodes]) for l in lines[1:]]) + '\n')
    d = tmp_path / 'node'
    os.makedirs(str(d))
    _launch([mesh, 'node', 'graph', 1, d / 'out', base], 2, cwd=str(d))
    node = [dict(np.load(str(d / 'out' / ('rank%d.npz' % r)))) for r in range(2)]
    _, files = run_files(tmp_path / 'files', mesh, 2, base)
    marked = 0
    for a, b in zip(files, node):
        def cutFaces(z):
            nodes = z['map.nodeNumbering_subdomain2global'][z['sub.elementBoundaryNodesArray']]
            faces = z['sub.elementBoundaryMaterialTypes'] == RANK_INTERFACE
            return sorted(tuple(sorted(f)) for f in nodes[faces].tolist())
        assert cutFaces(a) == cutFaces(b)
        exterior = a['sub.exteriorElementBoundariesArray']
        assert set(a['sub.elementBoundaryMaterialTypes'][exterior].tolist()) == {1, RANK_INTERFACE}
        marked += len(cutFaces(a))
    assert marked > 0
