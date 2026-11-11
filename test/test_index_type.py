"""The mesh index type has one definition, and C and Python agree on it.

``proteus_index_t`` in ``proteus/index_type.h`` is the type of every mesh
entity number, count and offset in ``struct Mesh``. ``cmeshTools.INDEX_DTYPE``
is read from it at compile time, and every index array a mesh hands to Python
must have that dtype.

This is C33's second step, and its third (slice 1) carries the same type
through the partitioner and the multilevel hierarchy. The width is still 32 bits; what these tests pin is
that it is decided in one place. Kernels and Python modules that still say
``int`` and ``'i'`` are converted a module at a time, each one adding its
arrays here.
"""
import os
import re

import numpy as np
import pytest

from proteus import Comm, MeshTools
from proteus import cmeshTools
from proteus.Archiver import GENERATED_INDEX_DTYPE

#: in the source tree, not next to the installed package: a non-editable
#: install (CI) does not ship the headers
HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      os.pardir, "proteus", "index_type.h")

#: numbers, counts and offsets of mesh entities: proteus_index_t
INDEX_ARRAYS = [
    "elementNodesArray",
    "nodeElementsArray",
    "nodeElementOffsets",
    "elementNeighborsArray",
    "elementBoundariesArray",
    "elementBoundaryNodesArray",
    "elementBoundaryElementsArray",
    "interiorElementBoundariesArray",
    "exteriorElementBoundariesArray",
    "edgeNodesArray",
    "nodeStarArray",
    "nodeStarOffsets",
]
PARTITION_ARRAYS = [
    "elementOffsets_subdomain_owned",
    "elementNumbering_subdomain2global",
    "nodeOffsets_subdomain_owned",
    "nodeNumbering_subdomain2global",
    "elementBoundaryOffsets_subdomain_owned",
    "elementBoundaryNumbering_subdomain2global",
    "edgeOffsets_subdomain_owned",
    "edgeNumbering_subdomain2global",
    "nodeNumbering_global_old2new",
    "elementNumbering_global_old2new",
]
#: local numbers and tags with a small range: a separate type, still int
LOCAL_ARRAYS = [
    "elementBoundaryLocalElementBoundariesArray",
    "elementMaterialTypes",
    "elementBoundaryMaterialTypes",
    "nodeMaterialTypes",
]


def test_python_reads_the_width_the_header_declares():
    with open(HEADER) as f:
        declared = re.search(r"typedef\s+(u?)int(\d+)_t\s+proteus_index_t\s*;",
                             f.read())
    assert declared, "index_type.h must declare proteus_index_t as a fixed-width type"
    unsigned, bits = declared.groups()
    assert cmeshTools.INDEX_DTYPE == np.dtype(("u" if unsigned else "i") + str(int(bits) // 8))


def test_the_archive_numbers_points_with_the_mesh_index_type():
    assert GENERATED_INDEX_DTYPE == cmeshTools.INDEX_DTYPE


@pytest.fixture(scope="module")
def partitioned_mesh():
    Comm.init()
    mlMesh = MeshTools.MultilevelTriangularMesh(
        4, 4, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, refinementLevels=1,
        nLayersOfOverlap=0,
        parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
    return mlMesh.meshList[-1]


def test_a_meshs_index_arrays_have_the_index_dtype(partitioned_mesh):
    for mesh in (partitioned_mesh, partitioned_mesh.subdomainMesh):
        for name in INDEX_ARRAYS:
            assert getattr(mesh, name).dtype == cmeshTools.INDEX_DTYPE, name
        for name in LOCAL_ARRAYS:
            assert getattr(mesh, name).dtype == np.intc, name


def test_the_partition_has_the_index_dtype(partitioned_mesh):
    for name in PARTITION_ARRAYS:
        assert getattr(partitioned_mesh, name).dtype == cmeshTools.INDEX_DTYPE, name


def test_the_partitioner_module_reads_the_same_width():
    from proteus import cpartitioning
    assert cpartitioning.INDEX_DTYPE == cmeshTools.INDEX_DTYPE


def test_the_node_graph_rows_have_the_index_dtype(partitioned_mesh):
    from proteus import cpartitioning
    offsets, neighbors = cpartitioning.nodeGraphRows(partitioned_mesh.cmesh, 0, 3)
    assert offsets.dtype == cmeshTools.INDEX_DTYPE
    assert neighbors.dtype == cmeshTools.INDEX_DTYPE


def test_the_file_readers_hand_back_index_arrays():
    from proteus import cpartitioning
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci", "meshPartition", "pslg")
    nElements, stars = cpartitioning.readNodeStars(base, 3, 0, 4, 0, 4)
    for star in stars:
        assert star.dtype == cmeshTools.INDEX_DTYPE
    _, _, elements, elementNodes, _ = cpartitioning.readSimplexElementsTouching(base, 3, 0, 4, [0])
    assert elements.dtype == cmeshTools.INDEX_DTYPE
    assert elementNodes.dtype == cmeshTools.INDEX_DTYPE


def test_the_multilevel_hierarchy_has_the_index_dtype():
    Comm.init()
    mlMesh = MeshTools.MultilevelTriangularMesh(
        4, 4, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, refinementLevels=2,
        nLayersOfOverlap=0,
        parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
    for arrays in (mlMesh.elementParentsArrayList[1:], mlMesh.elementChildrenArrayList,
                   mlMesh.elementChildrenOffsetsList):
        for a in arrays:
            assert a.dtype == cmeshTools.INDEX_DTYPE


def test_a_finite_element_spaces_dof_numbering_has_the_index_dtype(partitioned_mesh):
    from proteus import FemTools
    space = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(partitioned_mesh, 2)
    assert space.dofMap.l2g.dtype == cmeshTools.INDEX_DTYPE
    assert space.dofMap.subdomain2global.dtype == cmeshTools.INDEX_DTYPE
    assert space.dofMap.dof_offsets_subdomain_owned.dtype == cmeshTools.INDEX_DTYPE
    assert space.dofMap.l2g.shape[0] == partitioned_mesh.nElements_global


def test_the_sparsity_pattern_has_the_index_dtype():
    from proteus import csparsity
    n = cmeshTools.INDEX_DTYPE
    empty = np.zeros((0,), dtype=n)
    # one triangle with three free dofs: the pattern is the full 3x3 block
    info = csparsity.PySparsityInfo()
    info.findNonzeros(1, 3, 3, np.array([3], dtype=np.intc), np.arange(3, dtype=n).reshape(1, 3),
                      np.array([3], dtype=np.intc), np.arange(3, dtype=n).reshape(1, 3),
                      0, 1, 0, 1, 0, 0, 0, 3, np.full((1, 3), -1, dtype=n), 0, empty,
                      np.zeros((0, 2), dtype=n), np.zeros((0, 2), dtype=np.intc), 0, 0, empty, 0, 0)
    rowptr, colind, nnz, nzval = info.getCSR()
    assert rowptr.dtype == csparsity.CSR_INDEX_DTYPE
    assert colind.dtype == csparsity.CSR_INDEX_DTYPE
    assert list(rowptr) == [0, 3, 6, 9]


# --- the CSR index type (C33, decision 2026-11-10): its own type, = PetscInt ---

def test_the_csr_index_type_is_declared_and_exported():
    from proteus import csparsity, cpartitioning
    with open(HEADER) as f:
        text = f.read()
    # the width is a build setting (PROTEUS_CSR_INDEX_BITS, read from PETSc by setup.py)
    assert re.search(r"typedef\s+int64_t\s+proteus_csr_index_t", text)
    assert re.search(r"typedef\s+int32_t\s+proteus_csr_index_t", text)
    assert csparsity.CSR_INDEX_DTYPE.kind == "i"
    assert cpartitioning.CSR_INDEX_DTYPE == csparsity.CSR_INDEX_DTYPE


def test_the_csr_index_type_is_the_width_of_petscint():
    from proteus import csparsity
    PETSc = pytest.importorskip("petsc4py.PETSc")
    assert csparsity.CSR_INDEX_DTYPE == np.dtype(PETSc.IntType)


def test_the_csr_index_type_varies_independently_of_the_mesh_index_type():
    # no assertion on equality: the two are different questions (a mesh of
    # 3e9 nodes on a 32-bit PETSc is legal); each only has to be what it says
    from proteus import csparsity
    assert cmeshTools.INDEX_DTYPE.itemsize in (4, 8)
    assert csparsity.CSR_INDEX_DTYPE.itemsize in (4, 8)


@pytest.mark.parametrize("width", [np.int32, np.int64])
def test_superlu_takes_csr_arrays_of_either_width(width):
    # SuperLU's int is 32 bits; the wrapper converts the structure only when
    # the caller's width differs, and never touches the caller's arrays
    from proteus import superluWrappers
    rowptr = np.array([0, 2, 5, 8], dtype=width)
    colind = np.array([0, 1, 0, 1, 2, 0, 1, 2], dtype=width)
    nzval = np.arange(1.0, 9.0)
    A = superluWrappers.SparseMatrix(3, 3, 8, nzval, colind, rowptr)
    x = np.array([1.0, 2.0, 3.0])
    y = np.zeros(3)
    A.matvec(x, y)
    expected = np.array([1 * 1 + 2 * 2, 3 * 1 + 4 * 2 + 5 * 3, 6 * 1 + 7 * 2 + 8 * 3], dtype=float)
    assert np.allclose(y, expected)
    assert rowptr.dtype == width and colind.dtype == width


def test_a_matrix_too_large_for_superlu_is_refused():
    from proteus import superluWrappers
    nnz = 2 ** 31
    with pytest.raises(OverflowError):
        superluWrappers.SparseMatrix(1, 1, nnz, np.zeros(1), np.zeros(1, dtype=np.int64), np.zeros(2, dtype=np.int64))

