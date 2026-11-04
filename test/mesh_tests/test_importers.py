"""The in-tree mesh importers left after the SMS/ADH readers moved to ymf-import-sms (C7.1).

Each importer gets a small fixture and a test that pins the mesh it reads, per the
test-meshing policy. An importer that cannot read anything today is pinned with a strict
xfail naming why, so a fix shows up as an unexpected pass. The 3DM and ADH tests that were
here (proteus-written cube, square4x4, erdc/AdhModel's angle_bt, the CHL cases opt-in) now
run in proteustoolkit/ymf-import-sms against its converter.

- IFISS (``QuadrilateralMesh.generateFromQuadFileIFISS``): ``.mat`` grids made in the
  test, in the reader's own layout and in IFISS 3.7's.
- NURBS (``MultilevelNURBSMesh``): the box.
- DMPlex is round-tripped in ``test_generate_mesh_errors.py``.
"""
import os

import numpy as np
import pytest

from proteus import Comm, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))


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


@pytest.mark.xfail(strict=True,
                   reason="bdyflags are applied by file node number after finalize() has "
                          "renumbered the nodes, so the flags land on other nodes")
def test_the_ifiss_reader_puts_boundary_flags_on_their_nodes(tmp_path):
    """Each boundary edge gets the larger of its two nodes' flags.

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


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="IFISS 3.7 saves a grid as mv/xy/bound/mbound; the reader "
                          "expects vertices/quads/bdyflags, which no IFISS release "
                          "we have found writes")
def test_the_ifiss_reader_reads_an_ifiss_grid(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    scipy_io.savemat(str(tmp_path / "square_grid.mat"), _ifiss_usquare_grid(2))
    MeshTools.QuadrilateralMesh().generateFromQuadFileIFISS(str(tmp_path / "square_grid"))


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


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="cmeshTools lost generateNURBSMeshFromRectangularGrid in the "
                          "cmeshTools refactor (10689f82); mesh.cpp still has "
                          "regularNURBSMeshElements")
def test_a_nurbs_box_is_meshed():
    MeshTools.MultilevelNURBSMesh(3, 3, 3, 1, 1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  refinementLevels=1)
