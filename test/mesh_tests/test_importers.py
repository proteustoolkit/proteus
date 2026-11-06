"""The in-tree mesh importers left after the SMS/ADH readers moved to ymf-import-sms (C7.1).

Each importer gets a small fixture and a test that pins the mesh it reads, per the
test-meshing policy. An importer that cannot read anything today is pinned with a strict
xfail naming why, so a fix shows up as an unexpected pass. The 3DM and ADH tests that were
here (proteus-written cube, square4x4, erdc/AdhModel's angle_bt, the CHL cases opt-in) now
run in proteustoolkit/ymf-import-sms against its converter.

- IFISS (``QuadrilateralMesh.generateFromQuadFileIFISS``): ``.mat`` grids made in the
  test, in the reader's own layout and in IFISS 3.7's.
- NURBS (``MultilevelNURBSMesh``): deprecated (no function space uses it, it cannot build a mesh).
- DMPlex is round-tripped in ``test_generate_mesh_errors.py`` (a proteus mesh through a cell
  list), and here: ``gmsh_cube.msh`` read by PETSc, against the same file through msh2simplex.
- gmsh (``use_gmsh`` PSLG/PLC domains: ``msh2simplex``, then the triangle/tetgen readers):
  ``importer_meshes/gmsh_square.msh`` and ``gmsh_cube.msh``, written by gmsh 4.12.2 from the
  ``.geo`` files beside them (``Domain.writeGeo``, he_max 0.25 and 0.5). The stored ``.msh`` is
  the mesh the downstream tests read; meshing itself is checked on statistics.
- The ``-F`` regular grids (a box with ``generatePartitionedMeshFromFiles``): rank 0 writes
  the grid as triangle/tetgen files and every rank reads its part. Pinned against the box
  meshed in memory.

The solves on these meshes, serial against two ranks, are in ``test_importer_solves.py``.
"""
import os
import shutil

import numpy as np
import pytest

from proteus import Comm, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))
MESHES = os.path.join(HERE, "importer_meshes")


# --------------------------------------------------------------------------
# IFISS
# --------------------------------------------------------------------------

SIDES = {"bottom": 1, "right": 2, "top": 3, "left": 4}


def _square_quads(n=2):
    """The unit square in n x n quadrilaterals, 0-based, counterclockwise."""
    s = np.linspace(0.0, 1.0, n + 1)
    xy = np.array([[x, y] for y in s for x in s])
    quads = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1,
                       (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
                      for j in range(n) for i in range(n)])
    return xy, quads


def _ifiss_own_layout_square(tmp_path):
    """A 2x2 square in the reader's layout: ``vertices``, ``quads`` (0-based) and
    ``bdyflags`` rows (flag, node), each boundary node flagged with the larger
    flag of the sides it is on."""
    scipy_io = pytest.importorskip("scipy.io")
    xy, quads = _square_quads(2)
    flags = []
    for n, (x, y) in enumerate(xy):
        on = [f for side, f in SIDES.items()
              if {"bottom": y == 0, "right": x == 1, "top": y == 1, "left": x == 0}[side]]
        if on:
            flags.append([max(on), n])
    scipy_io.savemat(str(tmp_path / "square.mat"),
                     {"vertices": xy, "quads": quads, "bdyflags": np.array(flags)})
    mesh = MeshTools.QuadrilateralMesh()
    mesh.generateFromQuadFileIFISS(str(tmp_path / "square"))
    return mesh, xy, quads


def test_the_ifiss_reader_reads_its_own_layout(tmp_path):
    mesh, xy, quads = _ifiss_own_layout_square(tmp_path)
    assert (mesh.nNodes_global, mesh.nElements_global, mesh.nElementBoundaries_global,
            mesh.nExteriorElementBoundaries_global) == (9, 4, 12, 8)
    # the same points, renumbered (finalize sorts the nodes by x, then y)
    assert sorted(map(tuple, mesh.nodeArray[:, :2].tolist())) == sorted(map(tuple, xy.tolist()))
    assert sum(abs(np.linalg.det(np.array([[1, *mesh.nodeArray[a, :2]], [1, *mesh.nodeArray[b, :2]],
                                           [1, *mesh.nodeArray[c, :2]]])))
               for a, b, c, _ in mesh.elementNodesArray) == pytest.approx(1.0)


def test_the_ifiss_reader_puts_boundary_flags_on_their_nodes(tmp_path):
    """Each boundary edge gets the larger of its two nodes' flags. The flags were applied by
    file node number after finalize() had renumbered the nodes, so they landed on other nodes.

    So a corner node's flag spreads to both sides it touches: the bottom side,
    flagged 1 at every node, would have its two edges tagged 4 (from the left
    corner) and 2 (from the right one). That is why a ymf mesh tags facets, not
    nodes.
    """
    mesh, xy, quads = _ifiss_own_layout_square(tmp_path)
    exterior = mesh.exteriorElementBoundariesArray
    tags = {}
    for ebN in exterior:
        n0, n1 = mesh.elementBoundaryNodesArray[ebN]
        mid = 0.5 * (mesh.nodeArray[n0] + mesh.nodeArray[n1])
        tags[(round(mid[0], 2), round(mid[1], 2))] = int(mesh.elementBoundaryMaterialTypes[ebN])
    assert tags == {(0.25, 0.0): 4, (0.75, 0.0): 2, (1.0, 0.25): 2, (1.0, 0.75): 3,
                    (0.75, 1.0): 3, (0.25, 1.0): 4, (0.0, 0.75): 4, (0.0, 0.25): 4}


def _ifiss_usquare_grid(nc=2):
    """The grid IFISS 3.7's ``grids/usquare_domain.m`` saves as square_grid.mat.

    A transcription of that function (IFISS 3.7, Copyright (c) 2023 H. C. Elman,
    A. Ramage and D. J. Silvester, BSD 2-Clause licence,
    https://github.com/mcbssds/IFISS_download): Q2 macroelements ``mv`` (9 nodes,
    1-based), coordinates ``xy``, boundary nodes ``bound`` and boundary
    (element, side) pairs ``mbound``.
    """
    n = 2 ** nc
    np_ = n // 2
    x = y = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    mv = []
    kx = ky = 1
    for _ in range(np_):
        for _ in range(np_):
            mref = (n + 1) * (ky - 1) + kx
            mv.append([mref, mref + 2, mref + 2 * n + 4, mref + 2 * n + 2, mref + 1,
                       mref + n + 3, mref + 2 * n + 3, mref + n + 1, mref + n + 2])
            kx += 2
        ky += 2
        kx = 1
    mv = np.array(mv)
    k1 = np.flatnonzero(xy[:, 1] == 0) + 1
    k2 = np.flatnonzero((xy[:, 0] == 1) & (xy[:, 1] > 0)) + 1
    k3 = np.flatnonzero((xy[:, 1] == 1) & (xy[:, 0] < 1) & (xy[:, 0] > 0)) + 1
    k4 = np.flatnonzero((xy[:, 0] == 0) & (xy[:, 1] > 0)) + 1
    mbound = [[e + 1, side] for side, (col, ks) in enumerate(
        [(4, k1), (5, k2), (6, k3), (7, k4)], start=1) for e in range(len(mv)) if mv[e, col] in ks]
    bound = np.sort(np.concatenate([k1, k2, k3, k4]))
    return {"mv": mv.astype(float), "xy": xy, "bound": bound.astype(float).reshape(-1, 1),
            "mbound": np.array(mbound, dtype=float), "grid_type": 1.0, "outbc": 1.0,
            "x": x.reshape(-1, 1), "y": y.reshape(-1, 1)}


def test_the_ifiss_reader_reads_an_ifiss_grid(tmp_path):
    """IFISS 3.7 saves a grid as mv/xy/bound/mbound; the reader read only vertices/quads/bdyflags,
    which no IFISS release we have found writes (KeyError). Each 9-node macroelement is four
    bilinear quadrilaterals; the mbound sides are tagged 1 (IFISS's Dirichlet boundary)."""
    scipy_io = pytest.importorskip("scipy.io")
    grid = _ifiss_usquare_grid(2)
    scipy_io.savemat(str(tmp_path / "square_grid.mat"), grid)
    mesh = MeshTools.QuadrilateralMesh()
    mesh.generateFromQuadFileIFISS(str(tmp_path / "square_grid"))
    assert (mesh.nNodes_global, mesh.nElements_global, mesh.nElementBoundaries_global,
            mesh.nExteriorElementBoundaries_global) == (25, 16, 40, 16)
    assert sorted(map(tuple, mesh.nodeArray[:, :2].tolist())) == sorted(map(tuple, grid["xy"].tolist()))
    area = sum(abs(np.linalg.det(np.array([[1, *mesh.nodeArray[a, :2]], [1, *mesh.nodeArray[b, :2]],
                                           [1, *mesh.nodeArray[c, :2]]]))) for a, b, c, _ in mesh.elementNodesArray)
    assert area == pytest.approx(1.0)
    exterior = mesh.exteriorElementBoundariesArray
    assert set(mesh.elementBoundaryMaterialTypes[exterior]) == {1}
    assert set(mesh.elementBoundaryMaterialTypes[mesh.interiorElementBoundariesArray]) == {0}
    onBoundary = {tuple(xy) for xy in grid["xy"][grid["bound"].astype(int).reshape(-1) - 1].tolist()}
    assert {tuple(x) for x, f in zip(mesh.nodeArray[:, :2].tolist(), mesh.nodeMaterialTypes) if f == 1} == onBoundary


def test_an_ifiss_grid_without_a_dirichlet_side_leaves_it_untagged(tmp_path):
    """IFISS leaves the natural (Neumann) part of the boundary out of mbound: 0, untagged."""
    scipy_io = pytest.importorskip("scipy.io")
    grid = _ifiss_usquare_grid(2)
    grid["mbound"] = grid["mbound"][grid["mbound"][:, 1] != 2]          # the right side is natural
    scipy_io.savemat(str(tmp_path / "square_grid.mat"), grid)
    mesh = MeshTools.QuadrilateralMesh()
    mesh.generateFromQuadFileIFISS(str(tmp_path / "square_grid"))
    exterior = mesh.exteriorElementBoundariesArray
    mids = mesh.nodeArray[mesh.elementBoundaryNodesArray[exterior]].mean(axis=1)
    tags = mesh.elementBoundaryMaterialTypes[exterior]
    assert set(tags[np.isclose(mids[:, 0], 1.0)]) == {0}
    assert set(tags[~np.isclose(mids[:, 0], 1.0)]) == {1}


def test_a_mat_file_in_neither_layout_is_refused_by_name(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    scipy_io.savemat(str(tmp_path / "other.mat"), {"p": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="neither an IFISS grid"):
        MeshTools.QuadrilateralMesh().generateFromQuadFileIFISS(str(tmp_path / "other"))


def test_the_ifiss_grid_transcription_is_the_unit_square():
    """The fixture itself: 4x4 Q2 nodes of a 2x2 macroelement grid, every side tagged."""
    grid = _ifiss_usquare_grid(2)
    assert grid["xy"].shape == (25, 2) and grid["mv"].shape == (4, 9)
    corners = grid["xy"][grid["mv"][:, :4].astype(int) - 1]
    assert np.allclose(corners.min(axis=(0, 1)), 0.0) and np.allclose(corners.max(axis=(0, 1)), 1.0)
    assert sorted(set(grid["mbound"][:, 1].astype(int))) == [1, 2, 3, 4]
    assert len(grid["bound"]) == 16


# --------------------------------------------------------------------------
# NURBS
# --------------------------------------------------------------------------


def test_the_nurbs_mesh_is_deprecated():
    """Deprecated (cekees 2026-11-06): no function space uses NURBS elements, there is no NURBS reader,
    and it cannot build a mesh -- cmeshTools lost generateNURBSMeshFromRectangularGrid in the cmeshTools
    refactor (10689f82), so the constructor raises AttributeError after warning. Removed in C7.1 step G."""
    with pytest.warns(DeprecationWarning, match="MultilevelNURBSMesh is deprecated"):
        with pytest.raises(AttributeError, match="generateNURBSMeshFromRectangularGrid"):
            MeshTools.MultilevelNURBSMesh(3, 3, 3, refinementLevels=1)
    with pytest.warns(DeprecationWarning, match="NURBSMesh is deprecated"):
        MeshTools.NURBSMesh()


# --------------------------------------------------------------------------
# gmsh
# --------------------------------------------------------------------------

def _gmsh_domain(nd, name):
    """The domains the fixtures' .geo files were written from (Domain.writeGeo)."""
    from proteus import Domain
    if nd == 2:
        domain = Domain.PlanarStraightLineGraphDomain(
            vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], vertexFlags=[1, 2, 3, 4],
            segments=[[0, 1], [1, 2], [2, 3], [3, 0]], segmentFlags=[1, 2, 3, 4], name=name)
        domain.facets = [[[0, 1, 2, 3]]]
        domain.facetFlags = [1]
    else:
        from proteus import SpatialTools as st
        domain = Domain.PiecewiseLinearComplexDomain(name=name)
        st.Cuboid(domain, dim=[1.0, 1.0, 1.0], coords=[0.5, 0.5, 0.5])
        st.assembleDomain(domain)
    return domain


# the tag of each side of the unit square / cube: (axis, 0 or 1) -> tag
GMSH_SIDES = {2: {(1, 0): 1, (0, 1): 2, (1, 1): 3, (0, 0): 4},
              3: {(2, 0): 1, (1, 1): 2, (0, 1): 3, (1, 0): 4, (0, 0): 5, (2, 1): 6}}
GMSH_CASES = [(2, "gmsh_square", 0.25), (3, "gmsh_cube", 0.5)]


def _read_msh2(path):
    """Nodes and elements of a gmsh 2.2 ASCII file: (coordinates, {gmsh element type: count})."""
    lines = open(path).read().split("\n")
    i = lines.index("$Nodes")
    nodes = np.array([[float(w) for w in line.split()[1:]] for line in lines[i + 2:i + 2 + int(lines[i + 1])]])
    j = lines.index("$Elements")
    counts = {}
    for line in lines[j + 2:j + 2 + int(lines[j + 1])]:
        kind = int(line.split()[1])
        counts[kind] = counts.get(kind, 0) + 1
    return nodes, counts


def _sides(mesh, nd, faces):
    """For each face, the (axis, 0 or 1) sides of the unit box its centroid is on."""
    mids = mesh.nodeArray[mesh.elementBoundaryNodesArray[faces]].mean(axis=1)
    return [tuple((k, int(round(c[k]))) for k in range(nd) if min(abs(c[k]), abs(c[k] - 1.0)) < 1.0e-9)
            for c in mids]


def _gmsh_mesh(tmp_path, nd, name):
    shutil.copy(os.path.join(MESHES, name + ".msh"), str(tmp_path))
    domain = _gmsh_domain(nd, name)
    domain.geofile = name
    options = MeshTools.getMeshOptions(domain)
    options.use_gmsh = True
    options.genMesh = False
    return MeshTools._generateMesh(domain, options).meshList[0]


@pytest.mark.parametrize("nd, name, he", GMSH_CASES)
def test_msh2simplex_keeps_the_coordinates_gmsh_wrote(tmp_path, monkeypatch, nd, name, he):
    """It wrote six decimals (%f), moving nodes by up to 5e-7 (4.94e-7 square, 4.98e-7 cube)."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(os.path.join(MESHES, name + ".msh"), name + ".msh")
    MeshTools.msh2simplex(name, nd=nd)
    written = np.loadtxt(name + ".node", skiprows=1)
    gmsh_nodes, _ = _read_msh2(name + ".msh")
    np.testing.assert_array_equal(written[:, 1:1 + nd], gmsh_nodes[:, :nd])


MSH_SEGMENTS_INTO_A_NODE = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 7 1 1 2
2 1 2 7 1 3 2
3 1 2 8 2 3 4
4 1 2 8 2 1 4
5 2 2 1 1 1 2 3
6 2 2 1 1 1 3 4
$EndElements
"""


def test_msh2simplex_flags_a_node_that_only_ends_segments(tmp_path, monkeypatch):
    """Nodes 2 and 4 are the second node of both their segments. The second node's flag
    was written to the first node, so they kept flag 0 and nodes 1 and 3 were overwritten:
    [8, 0, 8, 0]."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into.msh").write_text(MSH_SEGMENTS_INTO_A_NODE)
    MeshTools.msh2simplex("into", nd=2)
    flags = np.loadtxt("into.node", skiprows=1)[:, -1].astype(int).tolist()
    assert flags == [7, 7, 7, 8]


@pytest.mark.parametrize("nd, name, he", GMSH_CASES)
def test_a_gmsh_domain_reads_its_stored_mesh(tmp_path, monkeypatch, nd, name, he):
    """A use_gmsh domain whose .msh exists: msh2simplex and the triangle/tetgen reader, no gmsh run."""
    monkeypatch.chdir(tmp_path)
    mesh = _gmsh_mesh(tmp_path, nd, name)
    gmsh_nodes, counts = _read_msh2(name + ".msh")
    np.testing.assert_array_equal(mesh.nodeArray[:, :nd], gmsh_nodes[:, :nd])
    assert mesh.nElements_global == counts[2 if nd == 2 else 4]
    exterior = mesh.exteriorElementBoundariesArray
    assert len(exterior) == (32 if nd == 2 else 540)
    sides = _sides(mesh, nd, exterior)
    assert all(len(s) == 1 for s in sides)
    assert [GMSH_SIDES[nd][s[0]] for s in sides] == mesh.elementBoundaryMaterialTypes[exterior].tolist()
    assert set(mesh.elementBoundaryMaterialTypes[mesh.interiorElementBoundariesArray]) == {0}
    assert set(mesh.elementMaterialTypes) == {1}


@pytest.mark.parametrize("nd, name, he", GMSH_CASES)
def test_gmsh_meshes_a_domain_within_its_size(tmp_path, monkeypatch, nd, name, he):
    """Meshing itself, on statistics: gmsh runs on the .geo; counts near the stored mesh's,
    every edge no longer than 1.5 he_max, every exterior face on one side and tagged with it."""
    if shutil.which("gmsh") is None:
        pytest.skip("no gmsh")
    monkeypatch.chdir(tmp_path)
    shutil.copy(os.path.join(MESHES, name + ".geo"), name + ".geo")
    domain = _gmsh_domain(nd, name)
    domain.geofile = name
    options = MeshTools.getMeshOptions(domain)
    options.use_gmsh = True
    options.genMesh = True
    mesh = MeshTools._generateMesh(domain, options).meshList[0]
    stored = {2: (98, 162), 3: (341, 1125)}[nd]
    assert 0.7 * stored[0] < mesh.nNodes_global < 1.3 * stored[0]
    assert 0.7 * stored[1] < mesh.nElements_global < 1.3 * stored[1]
    edges = mesh.nodeArray[mesh.edgeNodesArray]
    assert np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max() <= 1.5 * he
    exterior = mesh.exteriorElementBoundariesArray
    sides = _sides(mesh, nd, exterior)
    assert all(len(s) == 1 for s in sides)
    assert [GMSH_SIDES[nd][s[0]] for s in sides] == mesh.elementBoundaryMaterialTypes[exterior].tolist()


def test_a_gmsh_3d_domain_reads_its_stored_tetgen_files(tmp_path, monkeypatch):
    """GMSH_3D_Domain with genMesh False and its polyfile's .ele/.node/.face present: no meshing."""
    from proteus import Domain
    monkeypatch.chdir(tmp_path)
    source = MeshTools.TetrahedralMesh()
    source.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    source.writeTetgenFiles("flume", 1)
    domain = Domain.GMSH_3D_Domain(geofile="flume", name="flume")
    MeshTools.getMeshOptions(domain).genMesh = False
    mesh = MeshTools._generateMesh(domain, MeshTools.getMeshOptions(domain)).meshList[0]
    np.testing.assert_array_equal(mesh.nodeArray, source.nodeArray)
    assert mesh.nElements_global == source.nElements_global


def test_a_gmsh_3d_domain_without_gmsh2tetgen_says_so(tmp_path, monkeypatch):
    """GMSH_3D_Domain converts gmsh's medit file with gmsh2tetgen, which no proteus environment
    installs. Each step ignored its exit status, so the error was AssertionError: no mesh.ele
    (after mv failed on four files); and the old sequence removed mesh.ele before tetgen read it."""
    if shutil.which("gmsh") is None or shutil.which("gmsh2tetgen") is not None:
        pytest.skip("needs gmsh, and no gmsh2tetgen")
    from proteus import Domain
    monkeypatch.chdir(tmp_path)
    shutil.copy(os.path.join(MESHES, "gmsh_cube.geo"), "gmsh_cube.geo")
    domain = Domain.GMSH_3D_Domain(geofile="gmsh_cube", name="cube3d")
    options = MeshTools.getMeshOptions(domain)
    options.genMesh = True
    options.he = 0.5
    with pytest.raises(RuntimeError, match="exit status 127: gmsh2tetgen"):
        MeshTools._generateMesh(domain, options)
    assert os.path.exists("cube3d.mesh")                # gmsh ran


# --------------------------------------------------------------------------
# -F regular grids
# --------------------------------------------------------------------------

X0 = [0.5, -0.25, 0.125]


def _box_mesh(nd, nn, fromFiles, genMesh=True):
    from proteus import Domain
    domain = Domain.RectangularDomain(L=(1.0,) * nd, x=X0[:nd], name="box")
    options = domain.MeshOptions
    options.nnx = options.nny = options.nn = nn
    options.nnz = nn if nd == 3 else None
    options.triangleFlag = 1
    options.genMesh = genMesh
    return MeshTools._generateMesh(domain, options, generatePartitionedMeshFromFiles=fromFiles).meshList[0]


def _by_position(mesh, nd):
    x = mesh.subdomainMesh.nodeArray[:, :nd]
    return x[np.lexsort(x.T[::-1])]


@pytest.mark.parametrize("nd", [2, 3])
def test_a_regular_grid_from_files_is_the_box_meshed_in_memory(tmp_path, monkeypatch, nd):
    """-F on a box off the origin. The nodes were shifted after the partition, so the element
    and element-boundary barycenters stayed at the box at the origin (off by 0.5)."""
    monkeypatch.chdir(tmp_path)
    inMemory, fromFiles = _box_mesh(nd, 5, False), _box_mesh(nd, 5, True)
    np.testing.assert_array_equal(_by_position(fromFiles, nd), _by_position(inMemory, nd))
    for mesh in (inMemory, fromFiles):
        sub = mesh.subdomainMesh
        np.testing.assert_allclose(sub.elementBarycentersArray,
                                   sub.nodeArray[sub.elementNodesArray].mean(axis=1), atol=1.0e-15)
        np.testing.assert_allclose(sub.elementBoundaryBarycentersArray,
                                   sub.nodeArray[sub.elementBoundaryNodesArray].mean(axis=1), atol=1.0e-15)
    assert fromFiles.subdomainMesh.nElements_global == inMemory.subdomainMesh.nElements_global
    exterior = fromFiles.subdomainMesh.exteriorElementBoundariesArray
    assert (fromFiles.subdomainMesh.elementBoundaryMaterialTypes[exterior] > 0).all()


def test_a_tetgen_grid_written_by_one_run_is_read_by_the_next(tmp_path, monkeypatch):
    """-F with genMesh False reads the files a genMesh run wrote. They held the box at the
    origin and only the genMesh run shifted it, so the second run's mesh was at the origin."""
    monkeypatch.chdir(tmp_path)
    written = _by_position(_box_mesh(3, 5, True), 3)
    read = _by_position(_box_mesh(3, 5, True, genMesh=False), 3)
    np.testing.assert_array_equal(read, written)
    assert read.min(axis=0).tolist() == X0


@pytest.mark.parametrize("nd", [2, 3])
def test_a_regular_grid_from_files_keeps_every_digit(tmp_path, monkeypatch, nd):
    """Spacing 1/3: the triangle and tetgen writers (meshio.cpp) wrote six significant digits,
    so the -F nodes were 3.3e-7 from the in-memory box's."""
    monkeypatch.chdir(tmp_path)
    np.testing.assert_array_equal(_by_position(_box_mesh(nd, 4, True), nd),
                                  _by_position(_box_mesh(nd, 4, False), nd))



# --------------------------------------------------------------------------
# DMPlex: a gmsh file through PETSc
# --------------------------------------------------------------------------

def _plex_mesh(path):
    PETSc = pytest.importorskip("petsc4py.PETSc")
    from proteus import Domain
    plex = PETSc.DMPlex().createFromFile(path, interpolate=True, comm=PETSc.COMM_SELF)
    return MeshTools._generateMesh(Domain.DMPlexDomain(plex), MeshTools.MeshOptions(nd=3)).meshList[0]


def _faces_by_position(mesh):
    """{sorted node coordinates of a face: its tag}."""
    corners = mesh.nodeArray[mesh.elementBoundaryNodesArray]
    return {tuple(sorted(map(tuple, c.round(12).tolist()))): int(t)
            for c, t in zip(corners, mesh.elementBoundaryMaterialTypes)}


def test_a_gmsh_file_read_by_petsc_is_the_mesh_msh2simplex_reads(tmp_path, monkeypatch):
    """The DMPlex importer took its material types from the plex's cell-type label (the kind
    of polytope), so every face had tag 3, interior or not, and every tetrahedron 6. They
    are the plex's "Face Sets" and "Cell Sets" (gmsh's physical groups)."""
    monkeypatch.chdir(tmp_path)
    viaGmsh = _gmsh_mesh(tmp_path, 3, "gmsh_cube")
    viaPlex = _plex_mesh("gmsh_cube.msh")
    by_position = lambda x: x[np.lexsort(x.T[::-1])]  # noqa: E731
    np.testing.assert_array_equal(by_position(viaPlex.nodeArray), by_position(viaGmsh.nodeArray))
    for count in ("nElements_global", "nElementBoundaries_global", "nExteriorElementBoundaries_global",
                  "nEdges_global"):
        assert getattr(viaPlex, count) == getattr(viaGmsh, count), count
    assert _faces_by_position(viaPlex) == _faces_by_position(viaGmsh)
    assert set(viaPlex.elementMaterialTypes) == {1}


def test_a_plex_node_takes_a_tag_of_a_face_it_is_on(tmp_path, monkeypatch):
    """With no vertex sets, a boundary node takes the tag of the first tagged exterior face it
    is on; interior nodes stay 0."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(os.path.join(MESHES, "gmsh_cube.msh"), "gmsh_cube.msh")
    mesh = _plex_mesh("gmsh_cube.msh")
    exterior = mesh.exteriorElementBoundariesArray
    tagsOn = {}
    for ebN in exterior:
        for nN in mesh.elementBoundaryNodesArray[ebN]:
            tagsOn.setdefault(int(nN), set()).add(int(mesh.elementBoundaryMaterialTypes[ebN]))
    for nN, tag in enumerate(mesh.nodeMaterialTypes):
        if nN in tagsOn:
            assert tag in tagsOn[nN], nN
        else:
            assert tag == 0, nN
    assert len(tagsOn) == 341 - 69


def _cube_plex(PETSc):
    source = MeshTools.TetrahedralMesh()
    source.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    return source, PETSc.DMPlex().createFromCellList(
        3, source.elementNodesArray.astype(PETSc.IntType), source.nodeArray.copy(),
        interpolate=True, comm=PETSc.COMM_SELF)


def test_a_plex_marker_label_tags_the_boundary(tmp_path, monkeypatch):
    """A plex with only "marker" (a box made with -dm_plex_separate_marker 0, or markBoundaryFaces)."""
    PETSc = pytest.importorskip("petsc4py.PETSc")
    from proteus import Domain
    monkeypatch.chdir(tmp_path)
    source, plex = _cube_plex(PETSc)
    plex.createLabel("marker")
    plex.markBoundaryFaces("marker", 1)
    mesh = MeshTools._generateMesh(Domain.DMPlexDomain(plex), MeshTools.MeshOptions(nd=3)).meshList[0]
    assert set(mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray]) == {1}
    assert set(mesh.elementBoundaryMaterialTypes[mesh.interiorElementBoundariesArray]) == {0}
    assert mesh.nExteriorElementBoundaries_global == source.nExteriorElementBoundaries_global


def test_a_two_dimensional_plex_is_refused_by_name(tmp_path, monkeypatch):
    PETSc = pytest.importorskip("petsc4py.PETSc")
    from proteus import Domain
    monkeypatch.chdir(tmp_path)
    plex = PETSc.DMPlex().createFromCellList(
        2, np.array([[0, 1, 2], [0, 2, 3]], dtype=PETSc.IntType),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), interpolate=True, comm=PETSc.COMM_SELF)
    with pytest.raises(ValueError, match="reads tetrahedra in 3D"):
        MeshTools._generateMesh(Domain.DMPlexDomain(plex), MeshTools.MeshOptions(nd=3))
