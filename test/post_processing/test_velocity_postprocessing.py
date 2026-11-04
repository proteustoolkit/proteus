"""Velocity post-processing (PostProcessingTools) on a heterogeneous Poisson problem, serially and on two ranks.

The problem is vpp_poisson_2d_p: -div(a grad u) = f on the unit square, Dirichlet on three sides and the exact flux
on the fourth. Each run goes through vpp_partition_mpi.py in its own directory, which writes one row per owned
element: the conservation residual the post-processor stores, the same residual recomputed from the post-processed
velocity (source integral minus the outward flux through the faces), the face fluxes and the velocity at the element
quadrature points.

These properties are tested:

* A conservative variant balances every element: the recomputed residual is round-off.
* The optimized variants (pwl-opt, pwl-bdm-opt, and pwl-bdm2-opt with P2) need weak Dirichlet conditions, balance
  the elements whose face velocities they compute, and give the serial answer on two and three ranks.
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

The low-K variant (pwl-ib-fix-0) makes the faces into a low-conductivity region no-flux; it is tested on a block of
conductivity 1e-8 (VPP_PROBLEM=lowk).

The DG variants (dg: RT0, dg-bdm: BDM1) project the interior-penalty numerical flux of a DG P1 or P2 solution, which
balances every element by construction; they need only the element and its faces, so with an element partition and one
layer of overlap they give the serial answer on any number of ranks. dg-bdm stored face velocities that were not its
own until 2026-11-03: the C routine indexed the face and the shape functions wrongly, its wrapper passed the number of
face quadrature points as the number of shape functions, and with P2 it evaluated the linear representation with the
quadratic basis. Its face fluxes missed the numerical flux by 0.5 (P1) and 1.3 (P2, 25 x 25 nodes).

point-eval-gwvd keeps the velocity an LDG flux computes (VPP_SPACE=ldg); it is tested for that and for its refusals.
The LDG solve itself is wrong on this problem, so its accuracy test is a strict xfail.

Every run checks the linear solver's converged reason: a solve stopped at the iteration limit makes every variant
differ from serial (CG/ASM does on the interior-penalty systems, which are solved directly here).
"""
import json
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


def _run(ranks, cwd, variant, order, overlap=1, check=True, space="c0", nn=13, problem="smooth", **controls):
    os.makedirs(cwd, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VPP_VARIANT=variant, VPP_ORDER=str(order), VPP_OVERLAP=str(overlap),
               VPP_SPACE=space, VPP_NN=str(nn), VPP_PROBLEM=problem)
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
        assert _summary(cwd)["converged_reason"] > 0, "the linear solve did not converge: no variant can match serial"
        return numpy.loadtxt(os.path.join(str(cwd), "elements.csv"), delimiter=",")
    return out


def _summary(cwd):
    with open(os.path.join(str(cwd), "summary.json")) as f:
        return json.load(f)


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


OPTIMIZED = [("pwl-opt", 1), ("pwl-bdm-opt", 1), ("pwl-bdm2-opt", 2)]  # pwl-bdm2-opt needs a P2 solution


@pytest.mark.parametrize("variant,order", OPTIMIZED, ids=[v for v, _ in OPTIMIZED])
def test_an_optimized_variant_needs_weak_dirichlet_conditions(tmp_path, variant, order):
    """The -opt variants take the boundary flux from the model's face velocity, so they refuse strong conditions."""
    out = _run(1, tmp_path, variant, order, check=False, VPP_STRONG=True)
    assert out.returncode != 0
    assert "weak (Nitsche-type)" in out.stdout + out.stderr


@pytest.mark.parametrize("variant,order", OPTIMIZED, ids=[v for v, _ in OPTIMIZED])
def test_an_optimized_variant_balances_the_interior_elements(tmp_path, variant, order):
    """With weak conditions the -opt variants balance every element whose face velocities they compute.

    They leave the boundary flux to the model. The mprans models (RANS2P, RANS3PF) put the weak boundary flux into
    the face velocity; this problem's classic Transport puts a one-sided value there, so the elements in the boundary
    row miss by up to 0.5 while the 200 interior elements balance to round-off (4e-15 measured; 1e-14 with P2).
    """
    rows = _run(1, tmp_path, variant, order)
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


OPTIMIZED_PARALLEL = [("pwl-opt", 1, 2, 0), ("pwl-opt", 1, 2, 1), ("pwl-bdm-opt", 1, 2, 1),
                      ("pwl-bdm2-opt", 2, 2, 1), ("pwl-bdm2-opt", 2, 3, 0)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant,order,ranks,overlap", OPTIMIZED_PARALLEL,
                         ids=["%s-P%d-%dranks-overlap%d" % case for case in OPTIMIZED_PARALLEL])
def test_an_optimized_variant_in_parallel_is_the_serial_answer(tmp_path, variant, order, ranks, overlap):
    """The -opt variants solve the stars of owned nodes and sum the face corrections across ranks (ParVec)."""
    serial = _run(1, tmp_path / "1", variant, order)
    parallel = _run(ranks, tmp_path / "2", variant, order, overlap)
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


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_element_partitioning_without_overlap_is_refused(tmp_path):
    """Owned elements would have faces with no neighbour on the rank; this used to end every rank with exit()."""
    out = _run(2, tmp_path, "pwl", 1, overlap=0, check=False, VPP_ELEMENT_PARTITION=True)
    assert out.returncode != 0
    assert "conservativeFlux needs an overlap" in out.stdout + out.stderr


DG = [("dg", 1), ("dg", 2), ("dg-bdm", 1), ("dg-bdm", 2)]


@pytest.mark.parametrize("variant,order", DG, ids=["%s-P%d" % case for case in DG])
def test_a_dg_variant_balances_every_element(tmp_path, variant, order):
    """The face fluxes are the numerical flux, which balances each element: 1.6e-13 measured (2.9e-13 on 25 x 25)."""
    rows = _run(1, tmp_path, variant, order, space="dg")
    assert len(rows) == 288
    assert numpy.abs(rows[:, 3]).max() < 1e-12
    assert numpy.abs(rows[:, 2] - rows[:, 3]).max() < 1e-12


DG_PARALLEL = [("dg", 1, 2), ("dg", 2, 4), ("dg-bdm", 1, 4), ("dg-bdm", 2, 2)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("variant,order,ranks", DG_PARALLEL, ids=["%s-P%d-%dranks" % case for case in DG_PARALLEL])
def test_a_dg_variant_in_parallel_is_the_serial_answer(tmp_path, variant, order, ranks):
    """Element partition, one layer of overlap: every owned element has its neighbours. 4e-12 measured."""
    serial = _run(1, tmp_path / "1", variant, order, space="dg")
    parallel = _run(ranks, tmp_path / "2", variant, order, space="dg")
    assert numpy.abs(serial[:, :2] - parallel[:, :2]).max() < 1e-12
    assert numpy.abs(parallel[:, 3]).max() < 1e-12
    assert numpy.abs(serial[:, 4:] - parallel[:, 4:]).max() < 1e-10


@pytest.mark.parametrize("variant", ["dg", "dg-bdm"])
def test_a_dg_variant_with_p2_has_the_exact_face_fluxes(tmp_path, variant):
    """u = x^2 + y^2 is in DG P2, so the numerical flux is the exact one: 1.7e-13 measured (it missed by 0.45 with
    dg-bdm before its face velocities were fixed)."""
    _run(1, tmp_path, variant, 2, space="dg")
    assert _summary(tmp_path)["max_face_flux_error"] < 1e-11


# (variant, order, measured rate): the RT0 velocity is first order whatever the solution's order; BDM1 from the exact
# P2 flux is second order. Measured L2 errors on 7, 13, 25 nodes: dg P1 0.126, 0.0629, 0.0313; dg-bdm P2 0.0143,
# 0.00359, 0.000897 (dg-bdm P1 is first order too, at twice the RT0 error: 0.263, 0.132, 0.0662).
DG_RATES = [("dg", 1, 1.0), ("dg", 2, 1.0), ("dg-bdm", 1, 1.0), ("dg-bdm", 2, 2.0)]


@pytest.mark.parametrize("variant,order,rate", DG_RATES, ids=["%s-P%d" % case[:2] for case in DG_RATES])
def test_a_dg_variant_converges_at_its_rate(tmp_path, variant, order, rate):
    errors = []
    for nn in (7, 13, 25):
        _run(1, tmp_path / str(nn), variant, order, space="dg", nn=nn)
        errors.append(_summary(tmp_path / str(nn))["l2_velocity_error"])
    rates = numpy.log2(numpy.array(errors[:-1]) / numpy.array(errors[1:]))
    assert numpy.abs(rates - rate).max() < 0.1, (errors, rates)


LOW_K = 1.0e-8  # the block's conductivity in vpp_poisson_2d_p (VPP_PROBLEM=lowk)


def test_the_low_k_variant_gives_the_block_no_flux(tmp_path):
    """pwl-ib-fix-0 flags faces where the conductivity jumps by more than 1e4 to below 1e-5 and makes them no-flux.

    The exact flux through the block's 32 faces is of the order of its conductivity, 1e-8; pwl puts 0.037 through
    them, pwl-ib-fix-0 none. The price is an element imbalance of the same order (0.38 times the conductivity,
    measured at 1e-6, 1e-8 and 1e-10), since the flagged faces carry the Galerkin solution's small flux nowhere.
    """
    pwl = _run(1, tmp_path / "pwl", "pwl", 1, problem="lowk")
    fixed = _run(1, tmp_path / "fix", "pwl-ib-fix-0", 1, problem="lowk")
    assert _summary(tmp_path / "pwl")["n_block_faces"] == _summary(tmp_path / "fix")["n_block_faces"] == 32
    assert _summary(tmp_path / "pwl")["max_block_face_flux"] > 1e-2
    assert numpy.abs(pwl[:, 3]).max() < 1e-12
    assert _summary(tmp_path / "fix")["max_block_face_flux"] == 0.0
    assert numpy.abs(fixed[:, 3]).max() < LOW_K


LOW_K_PARALLEL = [(2, 0), (4, 0), (2, 1), (4, 1)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks,overlap", LOW_K_PARALLEL, ids=["%dranks-overlap%d" % case for case in LOW_K_PARALLEL])
def test_the_low_k_variant_in_parallel_is_the_serial_answer(tmp_path, ranks, overlap):
    """pwl-ib-fix-0 shares pwl's node stars and its two exchanges since 2026-11-03; it had its own copy of the stars
    without them, and differed from serial by 0.30 (2 ranks) and 0.44 (4 ranks) at overlap 0, and 7e-3 and 1.1e-2 at
    overlap 1, where the block faces cut by the partition also leaked (2.4e-2 at overlap 0).

    The answer matches serial to the conductivity's order, not to round-off (measured 2.7e-8 to 5.0e-8, 4.5 times the
    conductivity at 1e-6, 1e-8, 1e-10): its stars are solved unpinned, and a star the flagged faces split is a
    singular system whose parts do not balance, so the solution depends on the order of the star's elements.
    """
    serial = _run(1, tmp_path / "1", "pwl-ib-fix-0", 1, problem="lowk")
    parallel = _run(ranks, tmp_path / "2", "pwl-ib-fix-0", 1, overlap, problem="lowk")
    assert numpy.abs(serial[:, :2] - parallel[:, :2]).max() < 1e-12
    assert _summary(tmp_path / "2")["max_block_face_flux"] == 0.0
    assert numpy.abs(parallel[:, 3]).max() < LOW_K
    assert numpy.abs(serial[:, 4:] - parallel[:, 4:]).max() < 10 * LOW_K


def test_point_eval_gwvd_keeps_the_ldg_velocity_and_sets_no_face_velocities(tmp_path):
    """point-eval-gwvd post-processes nothing: the velocity is the LDG flux's mixed-form velocity, unchanged, and
    the face velocities stay zero, so the elements do not balance through them (no variant to compare in parallel:
    it does no communication)."""
    rows = _run(1, tmp_path, "point-eval-gwvd", 1, space="ldg", VPP_EVALUATE=True)
    summary = _summary(tmp_path)
    assert summary["max_velocity_minus_ldg"] == 0.0
    assert numpy.abs(rows[:, 4:7]).max() == 0.0
    assert "nothing in proteus sets them" in summary["evaluate"]


def test_point_eval_gwvd_needs_an_ldg_flux(tmp_path):
    """With any other flux the transport writes no velocity, and point-eval-gwvd would hand on zeros."""
    out = _run(1, tmp_path, "point-eval-gwvd", 1, space="dg", check=False)
    assert out.returncode != 0
    assert "LDG numerical flux" in out.stdout + out.stderr


@pytest.mark.xfail(strict=True, reason="the LDG solve is wrong: u misses x^2 + y^2 by 13 (P1, 13 x 13), Newton "
                                       "diverges from it, and identical runs differ in the fourth digit")
def test_point_eval_gwvd_velocity_is_accurate(tmp_path):
    """The LDG velocity should be within the RT0 error of -a grad u (0.063 here); measured 1.7e2."""
    _run(1, tmp_path, "point-eval-gwvd", 1, space="ldg")
    assert _summary(tmp_path)["l2_velocity_error"] < 0.2
