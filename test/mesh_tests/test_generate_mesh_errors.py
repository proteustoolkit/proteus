"""Meshing paths that were broken on main (C7.1 step C): each fails without its fix.

- ``generateMesh`` retried with the numerics module whenever meshing failed, so the
  first error was swallowed.
- ``MeshTetgenDomain`` read ``domain.polyfile``, which that class never sets.
- the DMPlex branch of ``_generateMesh`` under ``-F`` used ``fileprefix``, never
  assigned there: a DMPlex domain has no tetgen files to partition.
"""
import os
import types

import numpy as np
import pytest

from proteus import Comm, Domain, MeshTools

Comm.init()


def test_a_meshing_error_is_not_swallowed(monkeypatch):
    calls = []

    def failing(domain, meshOptions, generatePartitionedMeshFromFiles=False, numerics=None):
        calls.append(meshOptions)
        if len(calls) == 1:
            raise RuntimeError("the mesher failed")
        return "a mesh from the second try"

    monkeypatch.setattr(MeshTools, "_generateMesh", failing)
    domain = Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0])
    MeshTools.getMeshOptions(domain).setTriangleOptions("pq30Dena0.1")
    physics = types.SimpleNamespace(domain=domain, genMesh=True)
    numerics = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="the mesher failed"):
        MeshTools.generateMesh(physics, numerics)
    assert len(calls) == 1


def test_mesh_options_without_a_size_still_fall_back_to_the_numerics_module(monkeypatch):
    seen = []
    monkeypatch.setattr(MeshTools, "_generateMesh",
                        lambda domain, meshOptions, *args: seen.append(meshOptions) or "mesh")
    domain = Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0])
    physics = types.SimpleNamespace(domain=domain, genMesh=True)
    numerics = types.SimpleNamespace(nn=3)
    assert MeshTools.generateMesh(physics, numerics) == "mesh"
    assert seen == [numerics] and numerics.genMesh is True


def _cube_tetgen_files(prefix):
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    mesh.writeTetgenFiles(prefix, 1)
    return mesh


def test_a_tetgen_mesh_domain_reads_its_meshfile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = _cube_tetgen_files("cube")
    domain = Domain.MeshTetgenDomain(fileprefix="cube")
    options = MeshTools.MeshOptions(nd=3)
    options.genMesh = False
    mlMesh = MeshTools._generateMesh(domain, options)
    mesh = mlMesh.meshList[0]
    assert mesh.nNodes_global == source.nNodes_global
    assert mesh.nElements_global == source.nElements_global
    np.testing.assert_array_equal(mesh.nodeArray, source.nodeArray)


def test_a_dmplex_domain_under_F_says_it_has_no_files(tmp_path, monkeypatch):
    PETSc = pytest.importorskip("petsc4py.PETSc")
    monkeypatch.chdir(tmp_path)
    # hexahedra: a simplex box needs a tetrahedral generator, which not every PETSc has
    plex = PETSc.DMPlex().createBoxMesh([2, 2, 2], simplex=False, comm=PETSc.COMM_SELF)
    domain = Domain.DMPlexDomain(plex)
    options = MeshTools.MeshOptions(nd=3)
    with pytest.raises(ValueError, match="DMPlex domain has no tetgen files"):
        MeshTools._generateMesh(domain, options, generatePartitionedMeshFromFiles=True)


def test_a_dmplex_domain_gives_the_mesh_it_was_built_from(tmp_path, monkeypatch):
    """The DMPlex importer, round-tripped: a tetrahedral mesh handed to PETSc and read back."""
    PETSc = pytest.importorskip("petsc4py.PETSc")
    monkeypatch.chdir(tmp_path)
    source = MeshTools.TetrahedralMesh()
    source.generateTetrahedralMeshFromRectangularGrid(3, 3, 3, 1.0, 1.0, 1.0)
    # generateFromPlex reads faces and edges from the DAG, so the plex is interpolated
    plex = PETSc.DMPlex().createFromCellList(
        3, source.elementNodesArray.astype(PETSc.IntType), source.nodeArray.copy(),
        interpolate=True, comm=PETSc.COMM_SELF)
    mesh = MeshTools._generateMesh(Domain.DMPlexDomain(plex), MeshTools.MeshOptions(nd=3)).meshList[0]
    np.testing.assert_array_equal(mesh.nodeArray, source.nodeArray)
    # the same tetrahedra; the order of a tetrahedron's nodes is proteus's (node i
    # opposite face i), not the cell list's
    as_sets = lambda cells: sorted(map(tuple, np.sort(cells, axis=1).tolist()))  # noqa: E731
    assert as_sets(mesh.elementNodesArray) == as_sets(source.elementNodesArray)
    for count in ("nElementBoundaries_global", "nExteriorElementBoundaries_global",
                  "nEdges_global"):
        assert getattr(mesh, count) == getattr(source, count), count


@pytest.mark.parametrize("Mesh, reader, files", [
    (MeshTools.TriangularMesh, "generateFromTriangleFiles", "triangle"),
    (MeshTools.TetrahedralMesh, "generateFromTetgenFiles", "tetgen"),
])
def test_a_missing_mesh_file_is_an_error_not_an_empty_mesh(tmp_path, monkeypatch, Mesh, reader, files):
    """The readers returned a mesh with no nodes; the partitioner then failed far from the cause."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(IOError, match="could not read a %s mesh from nothere.node" % files):
        getattr(Mesh(), reader)("nothere", 1)


def test_a_missing_mesh_generator_is_an_error(tmp_path, monkeypatch):
    """writeMeshFiles ignored the shell's exit status, so a triangle not on PATH left no files."""
    monkeypatch.chdir(tmp_path)
    domain = Domain.PlanarStraightLineGraphDomain(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        segments=[[0, 1], [1, 2], [2, 3], [3, 0]], name="box")
    domain.writePoly("box")
    options = MeshTools.MeshOptions(nd=2)
    options.setTriangleOptions("VApq30Dena0.1")
    monkeypatch.setenv("PATH", str(tmp_path))       # no triangle here
    with pytest.raises(RuntimeError, match="mesh generator failed with exit status 127"):
        MeshTools.writeMeshFiles(domain, options)
