"""A gmsh ``.msh`` is read as a ymf mesh, and only so (C7.1 rounds 4 and 5).

``ymf-import-gmsh`` (proteustoolkit/ymf-import-gmsh) converts a ``.msh`` to the ymf mesh proteus reads.
Here, ``importer_meshes/gmsh_{square,cube}.ymf`` and ``.h5`` are that converter's output for the stored
``.msh`` files; the sha256 in their provenance says which ``.msh`` they are the conversion of, so a
``use_gmsh`` domain reads them without the converter, and a ``.msh`` that changed since is not
mistaken for them. ``gmsh_{square,cube}_reference.npz`` are the arrays ``msh2simplex`` and the
triangle/tetgen readers built from the same ``.msh`` (the path round 5 removed): the mesh read through
the ymf path is bit-equal to them. ``-F`` reads the ymf mesh whole and partitions it like any other.
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


def _reference(name):
    """The mesh the removed path (msh2simplex, tetgen -Vfeen in 3D, the triangle/tetgen readers) built
    from the stored ``.msh``: its arrays, stored before that path was removed (C7.1 round 5)."""
    return dict(np.load(os.path.join(MESHES, name + "_reference.npz")))


def _assert_same(mesh, reference):
    assert set(reference) == set(MESH_ARRAYS)
    for key in MESH_ARRAYS:
        np.testing.assert_array_equal(getattr(mesh, key), reference[key], err_msg=key)


@pytest.mark.parametrize("nd, name", CASES)
def test_the_stored_ymf_mesh_is_the_conversion_of_the_stored_msh(tmp_path, monkeypatch, nd, name):
    monkeypatch.chdir(tmp_path)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    assert MeshTools._gmshYmfCurrent(name)
    _change_msh(name + ".msh")
    assert not MeshTools._gmshYmfCurrent(name)


@pytest.mark.parametrize("nd, name", CASES)
def test_a_use_gmsh_domain_reads_its_ymf_mesh_bit_equal_to_the_reference(tmp_path, monkeypatch, nd, name):
    """No converter, no triangle/tetgen files, no tetgen run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)       # not installed
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    domain, options = _domain(nd, name)
    mesh = MeshTools._generateMesh(domain, options).meshList[0]
    assert not os.path.exists(name + ".ele") and not os.path.exists(name + ".node")
    _assert_same(mesh, _reference(name))


@pytest.mark.parametrize("nd, name", CASES)
def test_partitioned_reads_take_the_same_ymf_path(tmp_path, monkeypatch, nd, name):
    """-F (generatePartitionedMeshFromFiles) used to read triangle/tetgen files; now the ymf mesh."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5"))
    domain, options = _domain(nd, name)
    mesh = MeshTools._generateMesh(domain, options, True).meshList[0]
    assert not os.path.exists(name + ".ele") and not os.path.exists(name + ".node")
    _assert_same(mesh, _reference(name))


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("nd, name", CASES)
def test_a_msh_with_no_current_ymf_mesh_and_no_converter_names_the_package(tmp_path, monkeypatch, nd, name,
                                                                          changed):
    """No fallback: a missing ymf mesh, or one of an earlier .msh, with no converter, fails and says why."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ymf_import_gmsh", None)
    _copy(tmp_path, name, (".msh", ".ymf", ".h5") if changed else (".msh",))
    if changed:
        _change_msh(name + ".msh")
    domain, options = _domain(nd, name)
    with pytest.raises(ImportError, match="ymf-import-gmsh"):
        MeshTools._generateMesh(domain, options)
    assert not os.path.exists(name + ".ele") and not os.path.exists(name + ".node")


@pytest.mark.parametrize("nd, name", CASES)
def test_the_converter_makes_the_ymf_mesh_when_installed(tmp_path, monkeypatch, nd, name):
    pytest.importorskip("ymf_import_gmsh")
    monkeypatch.chdir(tmp_path)
    _copy(tmp_path, name, (".msh",))
    domain, options = _domain(nd, name)
    mesh = MeshTools._generateMesh(domain, options).meshList[0]
    assert os.path.exists(name + ".ymf") and not os.path.exists(name + ".ele")
    _assert_same(mesh, _reference(name))


@pytest.mark.parametrize("nd, name", CASES)
def test_the_mesh_plan_of_a_gmsh_domain_is_the_ymf_mesh(tmp_path, monkeypatch, nd, name):
    """What pro msh plans and writes: the .msh and the ymf mesh, no triangle/tetgen files."""
    domain, options = _domain(nd, name)
    plan = MeshTools.meshFiles(domain, options)
    assert plan["generator"] == "gmsh"
    assert plan["outputs"] == [name + e for e in (".msh", ".ymf", ".h5")] and plan["optional"] == []
    assert "msh2simplex" not in plan["command"] and "ymf-import-gmsh" in plan["command"]


@pytest.mark.parametrize("nd, name", CASES)
def test_pro_msh_writes_the_ymf_mesh(tmp_path, monkeypatch, nd, name):
    pytest.importorskip("ymf_import_gmsh")
    if shutil.which("gmsh") is None:
        pytest.skip("no gmsh")
    from proteus.cli import mesh as pro_mesh
    monkeypatch.chdir(tmp_path)
    _copy(tmp_path, name, (".geo",))
    domain, options = _domain(nd, name)
    options.genMesh = True
    plan = MeshTools.meshFiles(domain, options)
    pro_mesh.make(plan, domain, options)
    for ext in (".msh", ".ymf", ".h5"):
        assert os.path.exists(name + ext)
    assert not os.path.exists(name + ".ele")
    assert pro_mesh.state(plan)[0] == "current"
