"""
Streamlines of the post-processed velocity on the H(div) spaces (C10b): ``PostProcessingTools.traceStreamline``
on fields of the spaces whose streamlines are known.

* RT0 contains constants: a uniform flow (1, 0.5) traces a straight line, exactly, element after element.
* BDM1 contains every linear field: the rotation v = (-(y - 1/2), x - 1/2) is divergence free and its
  streamlines are circles about (1/2, 1/2); the radius is kept to the Runge-Kutta error.
Both on a triangle mesh whose elements the line crosses (the walk through the face neighbours is exercised).
"""
import numpy as np
import pytest

from proteus import Comm, FemTools, MeshTools
from proteus.PostProcessingTools import traceStreamline


def _mesh(n=9):
    m = MeshTools.TriangularMesh()
    m.generateTriangularMeshFromRectangularGrid(n, n, 1.0, 1.0)
    Comm.init()
    m.partitionMesh(nLayersOfOverlap=1, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
    return m.subdomainMesh


def _function(mesh, family, field):
    space = FemTools.Hdiv_AffineOnSimplex(mesh, 2, family)
    f = FemTools.FiniteElementFunction(space, dof=space.interpolate(field), dim_dof=1, name="v")
    return f


def test_uniform_flow_is_a_straight_line_in_rt0():
    mesh = _mesh()
    f = _function(mesh, "rt0", lambda X: np.broadcast_to(np.array([1.0, 0.5]), X.shape[:-1] + (2,)).copy())
    pts, els = traceStreamline(f, [0.05, 0.1], 0.05, 12)
    assert len(pts) == 13
    t = 0.05 * np.arange(13)
    assert np.abs(pts - (np.array([0.05, 0.1]) + t[:, None] * np.array([1.0, 0.5]))).max() < 1.0e-12
    assert len(set(els.tolist())) >= 8            # it crossed many elements


def test_rotation_is_a_circle_in_bdm1_and_leaving_the_mesh_stops_the_trace():
    mesh = _mesh(17)
    f = _function(mesh, "bdm1", lambda X: np.stack([-(X[..., 1] - 0.5), X[..., 0] - 0.5], axis=-1))
    h, n = 0.02, 150                              # 3 time units: about half a turn of the circle of radius 0.3
    pts, els = traceStreamline(f, [0.8, 0.5], h, n)
    assert len(pts) == n + 1
    radius = np.linalg.norm(pts - 0.5, axis=1)
    assert np.abs(radius - 0.3).max() < 1.0e-6, np.abs(radius - 0.3).max()
    # angle advances at rate 1
    ang = np.unwrap(np.arctan2(pts[:, 1] - 0.5, pts[:, 0] - 0.5))
    assert abs((ang[-1] - ang[0]) - h * n) < 1.0e-6
    # the circle of radius 0.6 leaves the unit square: the trace ends there
    out, _ = traceStreamline(f, [1.0, 0.5], h, 400)
    assert 1 < len(out) < 401
