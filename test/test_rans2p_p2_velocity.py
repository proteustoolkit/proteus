"""RANS2P and RANS2P2D with a P2 velocity space no longer read past the end of ``mesh_l2g``.

``calculateResidual`` of both kernels computed the position of every velocity dof as
``mesh_dof[3*mesh_l2g[eN*nDOF_v_test_element + i]]`` (for the ball velocity of the particles): ``mesh_l2g`` has
nDOF_mesh_trial_element (3 in 2D, 4 in 3D) entries per element, the velocity dof index runs to 6 (2D) or 10 (3D) with P2,
so the lookup left the array and, on some meshes, the process (SIGSEGV; the faulting line was found with lldb on a ``-g``
build: RANS2P2D.h:3182). With P1 velocity the index happens to stay in range, which hid it. The position is now the
element's own Lagrange node (vertex, or the midpoint of the edge the dof lives on: ``velocityDOFLocation``).

The case is the manufactured Navier-Stokes problem of test_rans2p_mixed_bc_mms.py with SPACE=p1p2 (P1 pressure, P2
velocity, undeclared setters), which crashed 6 of 6 runs at 13 nodes per side (2D) and 3 of 3 at 7 nodes per side (3D) before. The runs are in a subprocess (the
failure is a segfault); each must complete with errors that decrease under refinement, and two ranks must give the
serial errors.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_rans2p_mixed_bc_mms import MIXED2, MIXED3, MPIEXEC, _run  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cache = {}

    def go(ranks=1, **case):
        key = (ranks, tuple(sorted(case.items())))
        if key not in cache:
            cache[key] = _run(tmp_path_factory, ranks, SPACE="p1p2", **case)
        return cache[key]
    return go


@pytest.mark.parametrize("nn", [13, 17])
def test_p2_velocity_2d_runs_on_the_meshes_that_crashed(run, nn):
    r = run(ND=2, NN=nn, KINDS=MIXED2, DECLARE=0)
    assert r["space"] == "p1p2"
    for k in ("u0", "u1", "p"):
        assert 0.0 < r["L2"][k] < 1e-1, (k, r["L2"])


def test_p2_velocity_2d_converges(run):
    coarse, fine = run(ND=2, NN=13, KINDS=MIXED2, DECLARE=0), run(ND=2, NN=33, KINDS=MIXED2, DECLARE=0)
    for k in ("u0", "u1"):
        assert fine["L2"][k] < 0.3 * coarse["L2"][k], (k, coarse["L2"][k], fine["L2"][k])


def test_p2_velocity_3d_runs(run):
    r = run(ND=3, NN=7, KINDS=MIXED3, DECLARE=0)
    for k in ("u0", "u1", "u2", "p"):
        assert 0.0 < r["L2"][k] < 1e-1, (k, r["L2"])


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_p2_velocity_two_ranks_give_the_serial_errors_2d(run):
    s = run(ND=2, NN=13, KINDS=MIXED2, DECLARE=0)
    r = run(ranks=2, ND=2, NN=13, KINDS=MIXED2, DECLARE=0)
    for k in ("u0", "u1", "p"):
        assert r["L2"][k] == pytest.approx(s["L2"][k], rel=1e-9), (k, s["L2"][k], r["L2"][k])
