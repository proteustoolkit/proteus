"""Gauges on a partitioned mesh, under real MPI.

``gauges_partition_mpi.py`` runs point, line and line integral gauges at 1, 2
and 3 ranks; every value has a closed form (u = x + 10 y), so the answer does
not depend on the partition.  The partition is not reproducible from run to
run, so the multi-rank cases run several times.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges_partition_mpi.py")


def _mpiexec():
    """The launcher beside this interpreter, which matches its mpi4py (see
    test_archive_parallel_metadata.py for why not just any mpiexec)."""
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def run_ranks(ranks, tmp_path):
    env = dict(os.environ)
    # more ranks than cores on a small CI runner
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    cmd = [sys.executable, HELPER]
    if MPIEXEC:  # even for one rank (test_archive_parallel_metadata.run_ranks says why)
        cmd = [MPIEXEC, "-n", str(ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert result.returncode == 0, "%d ranks failed:\nSTDOUT:\n%s\nSTDERR:\n%s" % (
        ranks, result.stdout[-2000:], result.stderr[-4000:])
    lines = result.stdout.splitlines()
    observed = [int(l.split("=")[1]) for l in lines if l.startswith("OBSERVED_SIZE=")]
    # a launcher that does not match mpi4py starts N independent one-rank jobs
    assert observed == [ranks], "asked for %d ranks, ran with %s" % (ranks, observed)
    payload = [l for l in lines if l.startswith("GAUGE_RESULT=")]
    assert payload, "no result in:\n%s" % result.stdout[-2000:]
    return json.loads(payload[0].split("=", 1)[1])


def xs(result):
    return np.array([float(n.split("[")[1].split()[0]) for n in result["names"]])


def check(result):
    u = lambda x, y: x + 10 * y
    # points, the last two a roundoff outside the domain and snapped onto it
    npt.assert_allclose(result["points"]["values"],
                        [u(0.37, 0.22), u(0.5, 0.5), u(0.25, 0.75), u(0.5, 0), u(1, 0.5)], rtol=1e-12)
    # a line that leaves the mesh at both ends: points only inside, none repeated
    x = xs(result["line"])
    assert x.min() == 0 and x.max() == 1
    assert len(np.unique(np.round(x, 9))) == len(x)
    npt.assert_allclose(result["line"]["values"], u(x, 0.25), rtol=1e-12)
    npt.assert_allclose(result["integral"]["values"], [3.0], rtol=1e-12)
    # the vertical line crosses the cuts: integral of x + 10 y at x = 0.31 over y in [0, 1]
    npt.assert_allclose(result["vertical_integral"]["values"], [0.31 + 5.0], rtol=1e-12)
    # the diagonal of the unit cube: 14 points, none repeated by roundoff
    x = xs(result["diagonal_3d"])
    npt.assert_allclose(x, np.arange(14)/13.0, atol=1e-5)
    npt.assert_allclose(result["diagonal_3d"]["values"], 111*x, rtol=1e-4)
    # the Turek cylinder mesh: the hole is skipped
    x = xs(result["turek_line"])
    npt.assert_allclose([x.min(), x.max()], [0.0, 2.2], atol=1e-12)
    npt.assert_allclose([x[x < 0.2].max(), x[x > 0.2].min()], [0.1, 0.3], atol=1e-5)
    npt.assert_allclose(result["turek_line"]["values"], x + 2.0, atol=1e-4)
    npt.assert_allclose(result["turek_integral"]["values"], [6.82 - 0.44], rtol=1e-10)


def check_discontinuous(result, reference):
    """The gauges of a discontinuous field: one on each side of every face the line
    crosses, with the values of the elements either side, as in one rank."""
    names = result["dg_line"]["names"]
    sides = [n.rsplit("(", 1)[1] for n in names]
    # the line enters the mesh, then each segment's entry is followed by its exit
    assert sides == ["after)", "before)"] * (len(names) // 2)
    assert names == reference["dg_line"]["names"]
    npt.assert_allclose(result["dg_line"]["values"], reference["dg_line"]["values"], rtol=1e-12)
    # the field jumps at faces, so a gauge on each side is not an accident of the mesh
    values = np.array(result["dg_line"]["values"])
    assert np.any(values[1:-1:2] != values[2:-1:2])
    npt.assert_allclose(result["dg_integral"]["values"], reference["dg_integral"]["values"], rtol=1e-12)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("ranks, repeats", [(1, 1), (2, 4), (3, 3)])
def test_gauges_do_not_depend_on_the_partition(ranks, repeats, tmp_path):
    reference = run_ranks(1, tmp_path) if ranks > 1 else None
    for _ in range(repeats):
        result = run_ranks(ranks, tmp_path)
        check(result)
        check_discontinuous(result, reference or result)
