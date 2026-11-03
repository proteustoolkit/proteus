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
