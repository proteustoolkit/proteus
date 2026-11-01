"""mprans kernels with their own exterior loop on the faces between ranks.

An element partition with no overlap leaves the faces a rank shares with its
neighbour as exterior faces of its subdomain. A kernel that finds neither a
Dirichlet value nor a flux condition there takes its open-boundary branch,
which is one-sided -- outflow gets ``u b.n``, inflow gets none -- so the two
ranks' terms do not cancel. FluxBoundaryConditions specifies a zero advective
flux on those faces, which every model that reads its advective flux
dictionary then applies as a prescribed flux.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "vof_partition_mpi.py")
STEP_HELPER = os.path.join(HERE, "rans2p_step_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, helper=None, **overrides):
    """One run of vof_partition_mpi.py, with the partition pinned.

    -mat_partitioning_type current keeps the contiguous initial partition, the
    same every time (PETSc's default is not reproducible), and the linear
    solve is converged far enough that the answers can be compared to 1e-8.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1", PART_TYPE="element", NLAYERS="0",
               PETSC_OPTIONS="-mat_partitioning_type current -vof_ksp_rtol 1e-14 "
                             "-vof_ksp_atol 1e-14 -vof_ksp_type gmres")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    helper = helper or HELPER
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, helper]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    result = {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}
    result["interfaces"] = sum(
        int(dict(i.split("=") for i in l.split()[1:])["interfaces"])
        for l in out.stdout.splitlines() if l.startswith("RANK"))
    return result


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("model", ["VOF", "VOF3P", "VOS3P"])
def test_vof_on_an_element_partition_without_overlap_gives_the_serial_answer(model, tmp_path):
    """Mass used to leave through the cut: 2.9e-02 on two ranks against 4.5e-02 on one."""
    serial = _run(1, tmp_path, MODEL=model)
    assert serial["interfaces"] == 0
    parallel = _run(2, tmp_path, MODEL=model)
    assert parallel["interfaces"] > 0, parallel
    numpy.testing.assert_allclose(parallel["mass"], serial["mass"], rtol=1e-8)
    numpy.testing.assert_allclose(parallel["l2"], serial["l2"], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("model", ["VOF", "VOF3P", "VOS3P"])
def test_vof_with_one_layer_of_overlap_is_unchanged(model, tmp_path):
    """The control: with overlap the cut faces are ghost-layer faces and already harmless."""
    serial = _run(1, tmp_path, MODEL=model)
    parallel = _run(2, tmp_path, MODEL=model, NLAYERS=1)
    numpy.testing.assert_allclose(parallel["mass"], serial["mass"], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_rans2p_step_on_an_element_partition_without_overlap_gives_the_serial_answer(tmp_path):
    """RANS2P skips faces with material type below 1 in its own exterior loop, so the
    faces between ranks never reach its open-boundary branch. Not a given for the other
    kernels (VOF has no such guard); this keeps the guard honest."""
    pytest.importorskip("petsc4py")
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage("superlu_dist"):
        pytest.skip("needs superlu_dist")
    serial = _run(1, tmp_path, helper=STEP_HELPER)
    parallel = _run(2, tmp_path, helper=STEP_HELPER)
    assert parallel["interfaces"] > 0, parallel
    for key in ("p2", "ps", "u2", "us", "v2", "vs"):
        numpy.testing.assert_allclose(parallel[key], serial[key], rtol=1e-6, atol=1e-10, err_msg=key)


NCLS_HELPER = os.path.join(HERE, "ncls_partition_mpi.py")


def _ncls_residual(ranks, cwd, rd, helper=None, **extra):
    """A residual harness's result by node (summed over ranks), for the same state on any partition."""
    env = dict(os.environ, OMP_NUM_THREADS="1", RD=rd,
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in extra.items()})
    helper = helper or NCLS_HELPER
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, helper]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    total = {}
    for r in range(ranks):
        d = numpy.load(os.path.join(str(cwd), "res_rank%d_of%d.npz" % (r, ranks)))
        for xy, v in zip(numpy.round(d["xy"], 9), d["r"]):
            total[tuple(xy)] = total.get(tuple(xy), 0.0) + v
    return total


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("rd", ["same", "perturbed"])
def test_ncls_residual_on_two_ranks_is_the_serial_residual(rd, tmp_path):
    """With a redistanced field unlike u, the inflow term -flow*(u - rd_u) on the
    faces between ranks was applied once, uncancelled: 21 nodes on the cut off by
    1.2e-2 against a residual of 1.6e-2."""
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    serial = _ncls_residual(1, tmp_path / "1", rd)
    parallel = _ncls_residual(2, tmp_path / "2", rd)
    assert set(serial) == set(parallel)
    diff = max(abs(serial[k] - parallel[k]) for k in serial)
    assert diff < 1e-12, diff


KAPPA_HELPER = os.path.join(HERE, "kappa_partition_mpi.py")
DISSIPATION_HELPER = os.path.join(HERE, "dissipation_partition_mpi.py")


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("helper", [KAPPA_HELPER, DISSIPATION_HELPER], ids=["Kappa", "Dissipation"])
def test_kappa_and_dissipation_residual_on_two_ranks_is_the_serial_residual(helper, tmp_path):
    """Their open-boundary branch is one-sided; the zero flux on rank-interface
    faces cancels it. The control, without that entry, differs by 3.8e-2 on the cut."""
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _ncls_residual(1, tmp_path / "1", "same", helper=helper)
    parallel = _ncls_residual(2, tmp_path / "2", "same", helper=helper)
    assert max(abs(serial[k] - parallel[k]) for k in serial) < 1e-12
    control = _ncls_residual(2, tmp_path / "3", "same", helper=helper, NOZERO=1)
    assert max(abs(serial[k] - control[k]) for k in serial) > 1e-3


RANS3PF_HELPER = os.path.join(HERE, "rans3pf_partition_mpi.py")


def _rans3pf(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", PART_TYPE="element", NLAYERS="0", NSTEPS="3",
               PETSC_OPTIONS="-mat_partitioning_type current -rans2p_ksp_type preonly "
                             "-rans2p_pc_type lu -rans2p_pc_factor_mat_solver_type mumps "
                             "-phi_ksp_type gmres -phi_ksp_rtol 1e-14 -phi_ksp_atol 1e-14 "
                             "-phi_pc_type jacobi -phi_ksp_max_it 5000 -pressure_ksp_type preonly "
                             "-pressure_pc_type lu -pressure_pc_factor_mat_solver_type mumps")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, RANS3PF_HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return numpy.array([float(v) for v in line.split()[1:3]])


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_rans3pf_projection_scheme_on_two_ranks_gives_the_serial_answer(tmp_path):
    """Momentum flux on the rank cut: 2.4706e-03 on two ranks without the zero-flux entry,
    2.4886e-03 with it and on one rank."""
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    serial = _rans3pf(1, tmp_path / "1")
    parallel = _rans3pf(2, tmp_path / "2")
    numpy.testing.assert_allclose(parallel, serial, rtol=1e-8)
    control = _rans3pf(2, tmp_path / "3", NOZERO=1)
    assert abs(control[0] - serial[0]) > 1e-3 * abs(serial[0])


CLSVOF_HELPER = os.path.join(HERE, "clsvof_partition_mpi.py")


def _clsvof(ranks, cwd, **overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", PART_TYPE="node", NLAYERS="1", NSTEPS="4", CASE="2",
               PETSC_OPTIONS="-mat_partitioning_type current -clsvof_ksp_type preonly "
                             "-clsvof_pc_type lu -clsvof_pc_factor_mat_solver_type mumps")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    # even for one rank (test_archive_parallel_metadata.run_ranks says why)
    argv = [sys.executable, CLSVOF_HELPER]
    if MPIEXEC:
        argv = [MPIEXEC, "-n", str(ranks)] + argv
    return subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(cwd))


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_clsvof_on_a_partition_with_overlap_gives_the_serial_answer_and_without_is_refused(tmp_path):
    """CLSVOF's lumped projections are not summed across ranks: a partition without overlap
    gave a 3e-3 relative error in the first step, so it is refused."""
    for d in ("1", "2", "3"):
        (tmp_path / d).mkdir()
    out = _clsvof(1, tmp_path / "1")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    serial = [float(v) for v in [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0].split()[1:3]]
    out = _clsvof(2, tmp_path / "2")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    parallel = [float(v) for v in [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0].split()[1:3]]
    numpy.testing.assert_allclose(parallel, serial, rtol=1e-10)
    out = _clsvof(2, tmp_path / "3", PART_TYPE="element", NLAYERS="0")
    assert out.returncode != 0
    assert "CLSVOF in parallel needs a node partition or at least one layer of overlap" in out.stdout + out.stderr
