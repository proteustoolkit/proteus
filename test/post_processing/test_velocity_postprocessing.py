"""Velocity post-processing (PostProcessingTools) on a heterogeneous Poisson problem, serially and on two ranks.

The problem is vpp_poisson_2d_p: -div(a grad u) = f on the unit square, Dirichlet on three sides and the exact flux
on the fourth. Each run goes through vpp_partition_mpi.py in its own directory, which writes one row per owned
element: the conservation residual the post-processor stores, the same residual recomputed from the post-processed
velocity (source integral minus the outward flux through the faces), the face fluxes and the velocity at the element
quadrature points.

These properties are tested:

* A conservative variant balances every element: the recomputed residual is round-off.
* The optimized variants (pwl-opt, pwl-bdm-opt) need weak Dirichlet conditions, balance the elements whose face
  velocities they compute, and give the serial answer on two ranks.
* The local Larson-Niklasson variants (pwl, pwl-bdm, pwl-bdm2) give the serial answer on two ranks, at overlap 0
  and 1. Two things were wrong there, and overlap 2 hid both. Each face flux is corrected by the solutions of the
  local problems on the stars of its nodes, and a ghost node's star is cut by the partition; now the node's owner
  sends its star solution. And an owned element can have a face whose neighbour is not on the rank (flag -1): it was
  treated as a boundary face carrying the transport's subdomain flux; now its velocity comes from the owner of one of
  its ghost nodes, where the face is interior. Before, two ranks differed from serial by up to 2.6 in the face fluxes
  near the cut, with element residuals of 0.18. With P2 the two-rank run crashed: the star arrays were indexed by
  vertex dof number, which is the node number only in serial.

The global-correction variants (pwc, sun-rt0, sun-gs-rt0) solve one system per rank, so they are refused on more than
one rank.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "vpp_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, variant, order, overlap=1, check=True, **controls):
    os.makedirs(cwd, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VPP_VARIANT=variant, VPP_ORDER=str(order), VPP_OVERLAP=str(overlap))
    env.update({k: "1" for k, on in controls.items() if on})
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    argv = [sys.executable, HELPER] if ranks == 1 else [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=env, cwd=str(cwd))
    if check:
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        return numpy.loadtxt(os.path.join(str(cwd), "elements.csv"), delimiter=",")
    return out


CONSERVATIVE = [("pwl", 1, 1e-12), ("pwl-bdm", 1, 1e-12), ("pwc", 1, 1e-12), ("sun-rt0", 1, 1e-12),
                ("sun-gs-rt0", 1, 1e-5),  # Gauss-Seidel stops at its own tolerance: 1e-6 measured
                ("pwl", 2, 1e-12), ("pwl-bdm", 2, 1e-12), ("pwl-bdm2", 2, 1e-12)]


@pytest.mark.parametrize("variant,order,tolerance", CONSERVATIVE,
                         ids=["%s-P%d" % (v, o) for v, o, _ in CONSERVATIVE])
def test_a_conservative_variant_balances_every_element(tmp_path, variant, order, tolerance):
    rows = _run(1, tmp_path, variant, order)
    assert len(rows) == 288
    stored, recomputed = rows[:, 2], rows[:, 3]
    assert numpy.abs(recomputed).max() < tolerance
    if variant.startswith("pwl"):
        # pwc and the sun variants leave the residual of an intermediate stage in conservationResidual (up to 7e-3
        # here); only the PWL family stores the final one
        assert numpy.abs(stored - recomputed).max() < tolerance


@pytest.mark.parametrize("variant", ["pwl-opt", "pwl-bdm-opt"])
def test_an_optimized_variant_needs_weak_dirichlet_conditions(tmp_path, variant):
    """The -opt variants take the boundary flux from the model's face velocity, so they refuse strong conditions."""
    out = _run(1, tmp_path, variant, 1, check=False, VPP_STRONG=True)
    assert out.returncode != 0
    assert "weak (Nitsche-type)" in out.stdout + out.stderr


@pytest.mark.parametrize("variant", ["pwl-opt", "pwl-bdm-opt"])
def test_an_optimized_variant_balances_the_interior_elements(tmp_path, variant):
    """With weak conditions the -opt variants balance every element whose face velocities they compute.

    They leave the boundary flux to the model. The mprans models (RANS2P, RANS3PF) put the weak boundary flux into
    the face velocity; this problem's classic Transport puts a one-sided value there, so the elements in the boundary
    row miss by up to 0.5 while the 200 interior elements balance to round-off (4e-15 measured).
    """
    rows = _run(1, tmp_path, variant, 1)
    h = 1.0 / 12  # the mesh spacing (13 x 13 nodes)
    cx, cy, residual = rows[:, 0], rows[:, 1], numpy.abs(rows[:, 3])
    boundary_row = (cx < h) | (cx > 1 - h) | (cy < h) | (cy > 1 - h)
    assert (~boundary_row).sum() == 200
    assert residual[~boundary_row].max() < 1e-12
    assert residual[boundary_row].max() > 0.1  # fails when the classic Transport supplies the weak boundary flux


PARALLEL = [("pwl", 1, 0), ("pwl", 1, 1), ("pwl-bdm", 1, 0), ("pwl-bdm", 1, 1),
            ("pwl", 2, 1), ("pwl-bdm", 2, 1), ("pwl-bdm2", 2, 1)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant,order,overlap", PARALLEL,
                         ids=["%s-P%d-overlap%d" % case for case in PARALLEL])
def test_a_local_variant_on_two_ranks_is_the_serial_answer(tmp_path, variant, order, overlap):
    serial = _run(1, tmp_path / "1", variant, order)
    parallel = _run(2, tmp_path / "2", variant, order, overlap)
    assert numpy.abs(serial[:, :2] - parallel[:, :2]).max() < 1e-12  # the same elements, in the same order
    assert numpy.abs(parallel[:, 3]).max() < 1e-12
    assert numpy.abs(serial[:, 4:] - parallel[:, 4:]).max() < 1e-10


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant,overlap", [("pwl-opt", 0), ("pwl-opt", 1), ("pwl-bdm-opt", 1)])
def test_an_optimized_variant_on_two_ranks_is_the_serial_answer(tmp_path, variant, overlap):
    """The -opt variants solve the stars of owned nodes and sum the face corrections across ranks (ParVec)."""
    serial = _run(1, tmp_path / "1", variant, 1)
    parallel = _run(2, tmp_path / "2", variant, 1, overlap)
    assert numpy.abs(serial[:, :2] - parallel[:, :2]).max() < 1e-12
    assert numpy.abs(serial[:, 3:] - parallel[:, 3:]).max() < 1e-10


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("control", ["VPP_NO_STAR_EXCHANGE", "VPP_NO_CUT_FACE_EXCHANGE"])
def test_each_exchange_is_needed(tmp_path, control):
    """Without either exchange two ranks are not the serial answer (overlap 1, where both are active): measured
    1e-2 without the star exchange and 1.4 without the cut-face exchange."""
    serial = _run(1, tmp_path / "1", "pwl", 1)
    parallel = _run(2, tmp_path / "2", "pwl", 1, 1, **{control: True})
    assert numpy.abs(serial[:, 4:] - parallel[:, 4:]).max() > 1e-3
    assert numpy.abs(parallel[:, 3]).max() > 1e-4  # 7.4e-4 without the star exchange, 0.13 without the other


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant", ["pwc", "sun-rt0", "sun-gs-rt0"])
def test_a_global_variant_is_refused_on_two_ranks(tmp_path, variant):
    out = _run(2, tmp_path, variant, 1, check=False)
    assert out.returncode != 0
    assert "one rank" in out.stdout + out.stderr
