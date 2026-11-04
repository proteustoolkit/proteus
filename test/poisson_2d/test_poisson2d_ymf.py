"""test_2dm's problem through a converted mesh: the same solution as through the 2DM file (C7.1 step E).

``test/mesh_tests/ymf_meshes/square4x4`` is ``square4x4.3dm``/``.bc`` converted by ymf-import-sms.
Solving on it with a ``MeshYmfDomain`` must give the solution the deprecated ``Mesh2DMDomain``
gives on the source file, bit for bit, on however many ranks the test runs on.
"""
from importlib import reload
import os
import warnings

import numpy as np
import pytest

from proteus import Comm, Domain, NumericalSolution, default_n, default_s, default_so
from . import poisson_het_2d_p
from . import poisson_het_2d_c0pk_n

HERE = os.path.dirname(os.path.abspath(__file__))
YMF_SQUARE = os.path.join(os.path.dirname(HERE), "mesh_tests", "ymf_meshes", "square4x4")


@pytest.fixture
def shared_dir(tmp_path, monkeypatch):
    """One working directory for every rank: rank 0 writes files the others read."""
    comm = Comm.get().comm.tompi4py()
    path = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    monkeypatch.chdir(path)
    yield path
    comm.barrier()


def _solve(opts, name, set_domain):
    reload(poisson_het_2d_p)
    reload(poisson_het_2d_c0pk_n)
    p, n = poisson_het_2d_p, poisson_het_2d_c0pk_n
    p.x0 = (0.0, 0.0, 0.)
    p.L = (1., 1., 1.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        set_domain(p)
    n.nLevels = 3
    reload(default_so)
    so = default_so
    so.name = p.name = name + "pe" + repr(Comm.get().size())
    reload(default_s)
    so.sList = [default_s]
    opts.gatherArchive = True
    n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
    n.linearSolver = default_n.KSP_petsc4py
    n.multilevelLinearSolver = default_n.KSP_petsc4py
    n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(name)
    model = ns.modelList[0].levelModelList[-1]
    return model.mesh.nodeArray.copy(), model.u[0].dof.copy()


def test_a_converted_mesh_solves_as_the_2dm_file_does(opts, shared_dir):
    def from_2dm(p):
        p.meshfile = os.path.join(HERE, "square4x4")

    def from_ymf(p):
        p.domain = Domain.MeshYmfDomain(YMF_SQUARE)
        p.meshfile = None

    nodes_2dm, u_2dm = _solve(opts, "poisson_2d_2dm", from_2dm)
    nodes_ymf, u_ymf = _solve(opts, "poisson_2d_ymf", from_ymf)
    np.testing.assert_array_equal(nodes_ymf, nodes_2dm)
    np.testing.assert_array_equal(u_ymf, u_2dm)
    assert np.abs(u_ymf).max() > 0
