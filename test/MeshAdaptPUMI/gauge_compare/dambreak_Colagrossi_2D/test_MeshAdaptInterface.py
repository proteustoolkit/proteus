"""Interface-driven adaptation of the dambreak, serial and on two ranks.

test_MeshAdaptRestart.py uses the `pseudo` size input, which never adapts, so it
checks the restart machinery but not adaptation. Here the mesh really adapts
(about 1600 elements to 2800, whichever the rank count): the PUMI mesh is generated
(on every rank, by gathering the Proteus mesh on rank 0 and migrating it by the
Proteus partition), a baseline runs on it, and a run that adapts to the free
surface must keep the water in the tank: the equivalent water height at the four
gauge columns, at the baseline's last time, agrees to 2%.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _bin(name):
    return os.path.join(os.path.dirname(sys.executable), name)


def _parun(ranks, cwd, options, name):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    argv = [_bin("parun"), "-C", options, "-D", name, "dambreak_Colagrossi_so.py"]
    # One rank runs directly, not under mpiexec as the other one-rank launchers do
    # (test/test_archive_parallel_metadata.run_ranks): a serial run ends in PUMI's
    # reconstruction, which calls exit(0) without MPI_Finalize, and Open MPI's
    # launcher reports that as an abnormal termination with exit status 1.
    if ranks > 1:
        launcher = _bin("mpiexec") if os.path.exists(_bin("mpiexec")) else shutil.which("mpiexec")
        argv = [launcher, "-n", str(ranks)] + argv
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return out.stdout + out.stderr


@pytest.mark.parametrize("ranks", [1, 2])
def test_interface_adaptation_conserves_the_water(tmp_path, ranks):
    case = tmp_path / "case"
    shutil.copytree(HERE, case, ignore=shutil.ignore_patterns(
        "baseline", "adapt_*", "Reconstructed*", "*.log", "__pycache__", "mesh.*", "*.h5", "*.ymf"))
    base = "usePUMI=True fixedTimeStep=False"
    _parun(ranks, case, "gen_mesh=True adapt=0 " + base, "baseline")  # writes Reconstructed.*
    _parun(ranks, case, "gen_mesh=False adapt=0 " + base, "baseline")
    out = _parun(ranks, case, "gen_mesh=False adapt=1 sizeInputs='interface' " + base, "adapt_interface")

    # the global count after each (re)construction; a rank's own lines start with "Rank:"
    counts = [int(c) for c in re.findall(r"^(?!Rank)(?:: )?Number of elements (\d+)", out, flags=re.M)]
    assert counts, out[-1000:]
    assert max(counts) > 1.5 * counts[0], counts  # it adapted: the interface band is refined

    baseline = np.loadtxt(case / "baseline" / "vof_height.csv", delimiter=",", skiprows=1)
    adapted = np.loadtxt(case / "adapt_interface" / "vof_height.csv", delimiter=",", skiprows=1)
    t = baseline[-1, 0]
    for column in range(1, baseline.shape[1]):
        wanted = baseline[-1, column]
        got = np.interp(t, adapted[:, 0], adapted[:, column])
        assert got == pytest.approx(wanted, rel=0.02), (column, got, wanted)
