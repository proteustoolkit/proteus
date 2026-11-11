"""Which archived frames carry the solver state (--state-every), and what a hot start does from one that does not.

C17/C14 round 4. The solver state (the lagged shock capturing and subgrid error, the time history: what a continue
hot start needs to be the uninterrupted run) is some 20 times the size of one frame's solution at 1,536 ranks, and
was written with every frame (4.1 s a frame against 0.44 s on 1.9.x, Torino S0H30). It is now written at the final
step, at a checkpoint (--wall-limit) and with every N-th frame (--state-every N); the other frames carry the solution.
"""
import os
import re
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "step_retry_mpi.py")
DECK = "damBreakLevelSet.py"
LOG = "damBreakLevelSet.log"


def _env(**overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", DECK=DECK, LOGL="3")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def _run(cwd, ranks=1, **overrides):
    os.makedirs(str(cwd), exist_ok=True)
    env = _env(PETSC_OPTIONS="-mat_partitioning_type current", **overrides)
    cmd = [sys.executable, HELPER]
    if ranks > 1:
        cmd = [_mpiexec(), "-n", str(ranks)] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    return out


def _mpiexec():
    import shutil
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


def _frames_with_state(cwd, model="rans2p"):
    """The frames of ``model`` whose write included the solver state, from the log."""
    frames = []
    for line in open(os.path.join(str(cwd), LOG)):
        m = re.search(r"Archive: frame (\d+) of model %s at .*solver state in" % model, line)
        if m:
            frames.append(int(m.group(1)))
    return frames


def _frames(cwd, model="rans2p"):
    return [int(m.group(1)) for m in (re.search(r"Archive: frame (\d+) of model %s at " % model, l)
                                      for l in open(os.path.join(str(cwd), LOG))) if m]


def _worst(a, b):
    assert a.shape == b.shape and numpy.abs(a[:, :4] - b[:, :4]).max() < 1e-10
    return max(numpy.abs(a[a[:, 0] == k, 4] - b[b[:, 0] == k, 4]).max() / (numpy.abs(a[a[:, 0] == k, 4]).max() + 1e-300)
               for k in numpy.unique(a[:, 0]))


def _u(cwd):
    return numpy.loadtxt(os.path.join(str(cwd), "u.csv"), delimiter=",", skiprows=1)


def test_frames_carry_the_solution_only_and_the_final_one_the_state(tmp_path):
    out = _run(tmp_path / "default", FT=0.04, DTOUT=0.01)
    assert out.returncode == 0, out.stderr[-2000:]
    frames = _frames(tmp_path / "default")
    assert len(frames) >= 4
    assert _frames_with_state(tmp_path / "default") == [frames[-1]]


def test_state_every_n_writes_the_state_with_every_nth_frame_and_the_final_one(tmp_path):
    out = _run(tmp_path / "every", FT=0.04, DTOUT=0.01, STATEEVERY=2)
    assert out.returncode == 0, out.stderr[-2000:]
    frames = _frames(tmp_path / "every")
    assert _frames_with_state(tmp_path / "every") == sorted({f for f in frames if f % 2 == 0} | {frames[-1]})
    # every frame: the state before the run's end (this is the behaviour before --state-every)
    _run(tmp_path / "all", FT=0.04, DTOUT=0.01, STATEEVERY=1)
    assert _frames_with_state(tmp_path / "all") == frames


def test_a_hot_start_from_a_frame_without_the_state_restarts_from_its_solution(tmp_path):
    """The archive of a run to 0.04 holds the solver state of its last frame only. A continue hot start from the
    frame at 0.02 restarts from that solution, says so in the log, and is the --hotStartMode restart run."""
    ft = 0.06
    assert _run(tmp_path / "stop", FT=0.04, DTOUT=0.01).returncode == 0
    import shutil
    shutil.copytree(str(tmp_path / "stop"), str(tmp_path / "restart"))
    out = _run(tmp_path / "stop", FT=ft, DTOUT=0.01, HOTSTART=0.02)
    assert out.returncode == 0, out.stderr[-2000:]
    log = open(os.path.join(str(tmp_path / "stop"), LOG)).read()
    assert "carries the solution only" in log and "restarting its history from the restart state" in log
    warned = [l for l in out.stderr.splitlines() if "RuntimeWarning: hot start (continue)" in l]
    assert len(warned) == 1 and "carries the solution only" in warned[0]
    assert _run(tmp_path / "restart", FT=ft, DTOUT=0.01, HOTSTART=0.02, HOTSTARTMODE="restart").returncode == 0
    assert _worst(_u(tmp_path / "stop"), _u(tmp_path / "restart")) == 0.0


def test_a_hot_start_from_the_final_frame_is_still_exact(tmp_path):
    """The final frame's state is what the default writes, so stopping at t and hot starting from t continues the
    uninterrupted run bit for bit (test_hotstart_levelset), with no warning."""
    ft, t = 0.06, 0.04
    full = _run(tmp_path / "full", FT=ft, DTOUT=0.01)
    assert full.returncode == 0, full.stderr[-2000:]
    assert _run(tmp_path / "hot", FT=t, DTOUT=0.01).returncode == 0
    out = _run(tmp_path / "hot", FT=ft, DTOUT=0.01, HOTSTART=t)
    assert out.returncode == 0, out.stderr[-2000:]
    assert not [l for l in out.stderr.splitlines() if "RuntimeWarning: hot start" in l]
    assert _worst(_u(tmp_path / "full"), _u(tmp_path / "hot")) == 0.0
