"""A gmsh ``.msh`` is read as a ymf mesh (C7.1 round 4).

``ymf-import-gmsh`` (proteustoolkit/ymf-import-gmsh, private, like ymf-import-sms) converts a
``.msh`` to the ymf mesh proteus reads. Here, ``importer_meshes/gmsh_{square,cube}.ymf`` and ``.h5``
are that converter's output for the stored ``.msh`` files; the sha256 in their provenance says which
``.msh`` they are the conversion of, so a ``use_gmsh`` domain reads them without the converter, and
a ``.msh`` that changed since is not mistaken for them. The acceptance test: the mesh read this way is
bit-equal to the one ``msh2simplex`` and the triangle/tetgen readers build (the path it replaces).

Not covered here: ``-F`` (partitioned reads of the triangle/tetgen files), which still takes the
deprecated ``msh2simplex`` path.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import Comm, MeshTools

Comm.init()

HERE = os.path.dirname(os.path.abspath(__file__))
MESHES = os.path.join(HERE, "importer_meshes")
CASES = [(2, "gmsh_square"), (3, "gmsh_cube")]
MESH_ARRAYS = ("nodeArray", "elementNodesArray", "elementMaterialTypes", "nodeMaterialTypes",
               "elementBoundaryNodesArray", "exteriorElementBoundariesArray",
               "elementBoundaryMaterialTypes")


def _copy(tmp_path, name, suffixes):
    for suffix in suffixes:
        shutil.copy(os.path.join(MESHES, name + suffix), str(tmp_path))


def _change_msh(path):
    """The file with a comment section before its nodes: a different file, the same mesh."""
    text = open(path).read()
    open(path, "w").write(text.replace("$Nodes", "$Comments\nedited\n$EndComments\n$Nodes", 1))


def _gmsh_domain(nd, name):
    """The domains the stored .geo files were written from (Domain.writeGeo)."""
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


def _domain(nd, name):
    domain = _gmsh_domain(nd, name)
    domain.geofile = name
    options = MeshTools.getMeshOptions(domain)
    options.use_gmsh = True
    options.genMesh = False
    return domain, options


def _simplex_mesh(tmp_path, nd, name):
    """The path being replaced: msh2simplex, tetgen (3D), the triangle/tetgen reader."""
    base = str(tmp_path / ("simplex_" + name))
    shutil.copy(os.path.join(MESHES, name + ".msh"), base + ".msh")
    with pytest.warns(DeprecationWarning, match="msh2simplex is deprecated"):
        MeshTools.msh2simplex(base, nd=nd)
    if nd == 2:
        mesh = MeshTools.TriangularMesh()
        mesh.generateFromTriangleFiles(base, 1)
        return mesh
    if shutil.which("tetgen") is None:
        pytest.skip("no tetgen")
    subprocess.run(["tetgen", "-Vfeen", base + ".ele"], check=True, capture_output=True, timeout=120)
    for ext in ("ele", "node", "face"):
        os.replace("%s.1.%s" % (base, ext), "%s.%s" % (base, ext))
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateFromTetgenFiles(base, 1)
    return mesh


def _assert_same(a, b):
    for key in MESH_ARRAYS:
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)


@pytest.mark.parametrize("nd, name", CASES)
def test_the_stored_ymf_mesh_is_the_conversion_of_the_stored_msh(tmp_path, monkeypatch, nd, name):
    monkeypatch.chdir(tmp_path)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    assert MeshTools._gmshYmfCurrent(name)
    _change_msh(name + ".msh")
    assert not MeshTools._gmshYmfCurrent(name)


@pytest.mark.parametrize("nd, name", CASES)
def test_a_use_gmsh_domain_reads_its_ymf_mesh_bit_equal_to_msh2simplex(tmp_path, monkeypatch, nd, name):
    """The round 4 acceptance. No converter, no triangle/tetgen files, no tetgen run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)       # not installed
    reference = _simplex_mesh(tmp_path, nd, name)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    domain, options = _domain(nd, name)
    mesh = MeshTools._generateMesh(domain, options).meshList[0]
    assert not os.path.exists(name + ".ele") and not os.path.exists(name + ".node")
    _assert_same(mesh, reference)


@pytest.mark.parametrize("nd, name", CASES)
def test_a_msh_with_no_ymf_mesh_and_no_converter_takes_the_deprecated_path(tmp_path, monkeypatch, nd, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)
    _copy(tmp_path, name, (".msh",))
    domain, options = _domain(nd, name)
    with pytest.warns(DeprecationWarning, match="msh2simplex is deprecated"):
        mesh = MeshTools._generateMesh(domain, options).meshList[0]
    assert os.path.exists(name + ".ele") and not os.path.exists(name + ".ymf")
    _assert_same(mesh, _simplex_mesh(tmp_path, nd, name))


@pytest.mark.parametrize("nd, name", CASES)
def test_a_changed_msh_is_not_read_as_its_old_ymf_mesh(tmp_path, monkeypatch, nd, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    _change_msh(name + ".msh")
    domain, options = _domain(nd, name)
    with pytest.warns(DeprecationWarning, match="msh2simplex is deprecated"):
        MeshTools._generateMesh(domain, options)
    assert os.path.exists(name + ".ele")


@pytest.mark.parametrize("nd, name", CASES)
def test_the_converter_makes_the_ymf_mesh_when_installed(tmp_path, monkeypatch, nd, name):
    pytest.importorskip("ymf_import_gmsh")
    monkeypatch.chdir(tmp_path)
    _copy(tmp_path, name, (".msh",))
    domain, options = _domain(nd, name)
    mesh = MeshTools._generateMesh(domain, options).meshList[0]
    assert os.path.exists(name + ".ymf") and not os.path.exists(name + ".ele")
    _assert_same(mesh, _simplex_mesh(tmp_path, nd, name))
