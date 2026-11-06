"""cmeshTools.generateFromNumberedArrays: a subdomain rebuilt from its arrays with its own numbering (C7.1 Phase D).

partitioning.cpp builds each subdomain from its nodes, elements, each element's boundary numbers and its
edges, through the constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_* and
geometric-info routines; generateFromNumberedArrays calls the same, so a rebuilt subdomain is the
partitioner's, numbering included. generateFromArrays numbers the boundaries and edges its own way.
"""
import os

import numpy as np
import pytest

from proteus import Comm, MeshTools, cmeshTools

Comm.init()
HERE = os.path.dirname(os.path.abspath(__file__))


def _subdomain(prefix, rank=0, size=2):
    """Rank ``rank`` of ``size``'s subdomain, partitioned in this process by C32's partitionNodesLocal
    (reversed contiguous owners), whatever the number of ranks the test runs on."""
    from proteus import cpartitioning
    mesh = MeshTools._meshFromYmf(os.path.join(HERE, "ymf_meshes", prefix))
    n = mesh.nNodes_global
    sub = mesh.__class__()
    sub.cmesh = cmeshTools.CMesh()
    cpartitioning.partitionNodesLocal(rank, size, 1, mesh.cmesh, sub.cmesh,
                                      (size - 1 - np.arange(n) * size // n).astype(np.intc))
    sub.buildFromC(sub.cmesh)
    return sub


def _rebuilt(sub, numbered=True):
    mesh = sub.__class__()
    if numbered:
        mesh.cmesh = cmeshTools.CMesh()
        cmeshTools.generateFromNumberedArrays(mesh.cmesh, sub.nodeArray, sub.elementNodesArray,
                                              sub.elementBoundariesArray, sub.edgeNodesArray,
                                              sub.nodeMaterialTypes, sub.elementMaterialTypes,
                                              sub.elementBoundaryMaterialTypes)
        mesh.buildFromC(mesh.cmesh)
    else:
        mesh.generateFromArrays(np.asarray(sub.nodeArray), np.asarray(sub.elementNodesArray),
                                np.asarray(sub.nodeMaterialTypes), np.asarray(sub.elementMaterialTypes))
    return mesh


def _differ(a, b):
    return [k for k, v in vars(a).items()
            if isinstance(v, np.ndarray) and hasattr(b, k) and not np.array_equal(np.asarray(v), np.asarray(getattr(b, k)))]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("prefix", ["square4x4_tagged", "cube_3x3x3"])
def test_a_subdomain_rebuilt_with_its_numbering_is_the_subdomain(prefix, rank):
    sub = _subdomain(prefix, rank)
    rebuilt = _rebuilt(sub)
    assert len([v for v in vars(sub).values() if isinstance(v, np.ndarray)]) >= 30
    assert _differ(sub, rebuilt) == []


@pytest.mark.parametrize("prefix", ["square4x4_tagged", "cube_3x3x3"])
def test_the_numbering_is_what_differs_from_generateFromArrays(prefix):
    sub = _subdomain(prefix)
    renumbered = _rebuilt(sub, numbered=False)
    assert np.array_equal(renumbered.elementNodesArray, sub.elementNodesArray)
    assert {"elementBoundariesArray", "elementBoundaryNodesArray", "edgeNodesArray"} <= set(_differ(sub, renumbered))


def test_mismatched_arrays_are_refused():
    sub = _subdomain("square4x4_tagged")
    with pytest.raises(ValueError, match="element boundary material types"):
        cmeshTools.generateFromNumberedArrays(cmeshTools.CMesh(), sub.nodeArray, sub.elementNodesArray,
                                              sub.elementBoundariesArray, sub.edgeNodesArray, sub.nodeMaterialTypes,
                                              sub.elementMaterialTypes, sub.elementBoundaryMaterialTypes[:-1])
    with pytest.raises(ValueError, match="as many boundaries as nodes"):
        cmeshTools.generateFromNumberedArrays(cmeshTools.CMesh(), sub.nodeArray, sub.elementNodesArray,
                                              np.asarray(sub.elementBoundariesArray)[:, :2], sub.edgeNodesArray,
                                              sub.nodeMaterialTypes, sub.elementMaterialTypes,
                                              sub.elementBoundaryMaterialTypes)
