"""Which archived frames carry the solver state (--state-every), and what a hot start does from one that does not.

C17/C14 round 4. The solver state (the lagged shock capturing and subgrid error, the time history: what a continue
hot start needs to be the uninterrupted run) is some 20 times the size of one frame's solution at 1,536 ranks, and
was written with every frame (4.1 s a frame against 0.44 s on 1.9.x, Torino S0H30). It is now written at the final
step, at a checkpoint (--wall-limit) and with every N-th frame (--state-every N); the other frames carry the solution.
"""
import glob
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


def _steps(cwd):
    return numpy.atleast_1d(numpy.loadtxt(os.path.join(str(cwd), "accepted.txt")))


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_run_past_its_wall_limit_stops_at_a_step_boundary_with_the_checkpoint_status(tmp_path, ranks):
    """WALLSTEPS=3 (the helper's clock counts accepted system steps, one second each, against a limit of 3.5 s:
    another step and a checkpoint do not fit after the third): exit status 75 on every rank, three steps taken."""
    if ranks > 1 and _mpiexec() is None:
        pytest.skip("needs mpiexec")
    out = _run(tmp_path, ranks=ranks, FT=0.05, DTOUT=0.01, WALLSTEPS=3)
    assert out.returncode == 75, out.stdout[-1500:] + out.stderr[-1500:]
    assert len(_steps(tmp_path)) == 3
    log = open(os.path.join(str(tmp_path), LOG)).read()
    assert "Wall limit: " in log and "stopping at t=" in log and "Wall limit: checkpoint at t=" in log
    from proteus import WallLimit
    assert WallLimit.CHECKPOINT_EXIT_STATUS == 75
    # the marker a job script tests when the launcher's exit status is not 75 (aprun under PALS: 143)
    markers = glob.glob(os.path.join(str(tmp_path), "*.checkpoint"))
    assert len(markers) == 1 and not glob.glob(os.path.join(str(tmp_path), "*.checkpoint.tmp*"))
    marker = WallLimit.read_marker(markers[0])
    assert marker["ranks"] == ranks and marker["exit_status"] == 75 and marker["frame"] == max(_frames(tmp_path))
    assert marker["t"] > 0.0 and "wrote %s" % os.path.basename(markers[0]) in log.replace("./", "")


def test_the_marker_is_gone_when_the_next_leg_finishes_and_never_written_by_a_finished_run(tmp_path):
    """A leg that stops leaves the marker; the leg after it (still under --wall-limit, never reached) removes it at
    its start and leaves none at its end. A run without a limit writes none, and does not touch one."""
    assert _run(tmp_path, FT=0.05, DTOUT=0.01, WALLSTEPS=3).returncode == 75
    marker = glob.glob(os.path.join(str(tmp_path), "*.checkpoint"))
    assert len(marker) == 1
    out = _run(tmp_path, FT=0.05, DTOUT=0.01, HOTSTART=1, WALLLIMIT="99:00:00")
    assert out.returncode == 0, out.stderr[-2000:]
    assert not glob.glob(os.path.join(str(tmp_path), "*.checkpoint"))
    done = _run(tmp_path / "done", FT=0.03, DTOUT=0.01, WALLLIMIT="99:00:00")
    assert done.returncode == 0 and not glob.glob(os.path.join(str(tmp_path / "done"), "*.checkpoint"))


def test_a_run_inside_its_wall_limit_is_the_run_without_one(tmp_path):
    """The limit is never reached (99 h): same exit status, steps and dofs as without it, and the same writes (the
    solver state of the final frame only: a limit does not add a state write)."""
    full = _run(tmp_path / "full", FT=0.04, DTOUT=0.01)
    limited = _run(tmp_path / "limited", FT=0.04, DTOUT=0.01, WALLLIMIT="99:00:00")
    assert full.returncode == 0 and limited.returncode == 0, limited.stderr[-2000:]
    numpy.testing.assert_array_equal(_steps(tmp_path / "full"), _steps(tmp_path / "limited"))
    assert _worst(_u(tmp_path / "full"), _u(tmp_path / "limited")) == 0.0
    assert _frames_with_state(tmp_path / "limited") == _frames_with_state(tmp_path / "full") == [_frames(tmp_path / "full")[-1]]


def test_the_checkpoint_cost_is_estimated_until_a_state_write_has_been_timed(tmp_path):
    """With no --state-every the state has not been written when the first step boundaries are checked: the cost
    of a checkpoint then includes an estimate (the frame's time times state bytes over dof bytes), so the margin
    is not frame-only. After --state-every writes it is the measured time."""
    from proteus.WallLimit import WallClock  # noqa: F401  (the margin itself is tested in test_wall_limit)
    out = _run(tmp_path, FT=0.05, DTOUT=0.01, WALLSTEPS=3, WALLCOST="measured")
    assert out.returncode == 75
    cost = [float(m.group(1)) for m in re.finditer(r"and a checkpoint \(([\d.]+) s\)",
                                                    open(os.path.join(str(tmp_path), LOG)).read())]
    assert cost and cost[-1] > 0.0


def test_a_malformed_wall_limit_stops_parun_before_it_runs(tmp_path):
    out = _run(tmp_path, FT=0.04, DTOUT=0.01, WALLLIMIT="soon")
    assert out.returncode == 2 and "wall limit" in out.stderr, out.stderr[-1500:]


FT, DTOUT = 0.1, 0.02


def _legs(tmp_path, stop_after, ranks=1, **slow):
    """The uninterrupted run, and the same run in two legs: the first stopped by its wall limit after
    ``stop_after`` accepted system steps (exit status 75), the second a hot start of its archive to the end."""
    full = _run(tmp_path / "full", ranks=ranks, FT=FT, DTOUT=DTOUT)
    assert full.returncode == 0, full.stderr[-2000:]
    first = _run(tmp_path / "legs", ranks=ranks, FT=FT, DTOUT=DTOUT, WALLSTEPS=stop_after, **slow)
    assert first.returncode == 75, first.stdout[-1500:] + first.stderr[-1500:]
    # stopped after exactly the steps asked for, whatever the machine's speed (a hosted runner's archive writes
    # made the measured checkpoint cost exceed the half step of slack, and the run stopped a step early)
    assert len(_steps(tmp_path / "legs")) == stop_after
    # the next leg writes the same log file
    first.log = open(os.path.join(str(tmp_path / "legs"), LOG)).read()
    second = _run(tmp_path / "legs", ranks=ranks, FT=FT, DTOUT=DTOUT, HOTSTART="inf", **slow)
    assert second.returncode == 0, second.stderr[-2000:]
    assert not [l for l in second.stderr.splitlines() if "RuntimeWarning: hot start" in l], second.stderr[-3000:]
    return full, first, second


def _same_run(tmp_path, ranks, stop_after):
    """Bit for bit: every rank's dofs, the steps the second leg takes, and the rows of the force histories."""
    full_steps = _steps(tmp_path / "full")
    assert len(full_steps) > stop_after
    # the second leg's accepted steps are the uninterrupted run's after the stop
    stop_time = full_steps[stop_after - 1]
    numpy.testing.assert_array_equal(_steps(tmp_path / "legs"), full_steps[full_steps > stop_time + 1e-12])
    for rank in range(ranks):
        name = "u.csv" if ranks == 1 else "u%d.csv" % rank
        a, b = (numpy.loadtxt(os.path.join(str(tmp_path / d), name), delimiter=",", skiprows=1) for d in ("full", "legs"))
        assert _worst(a, b) == 0.0
    for rows in ("forceHistory_p.txt", "forceHistory_v.txt", "momentHistory.txt"):
        numpy.testing.assert_array_equal(numpy.loadtxt(os.path.join(str(tmp_path / "legs"), rows)),
                                         numpy.loadtxt(os.path.join(str(tmp_path / "full"), rows)))


def _stop_steps(tmp_path):
    """Steps to stop after: in the middle of an interval, and exactly at the end of one (an archived frame)."""
    out = _run(tmp_path / "probe", FT=FT, DTOUT=DTOUT)
    assert out.returncode == 0, out.stderr[-2000:]
    steps = _steps(tmp_path / "probe")
    on_grid = [i + 1 for i, t in enumerate(steps) if abs(t / DTOUT - round(t / DTOUT)) < 1e-9 and t > 0.03][0]
    mid = [i + 1 for i, t in enumerate(steps) if t > 0.025 and abs(t / DTOUT - round(t / DTOUT)) > 1e-3][0]
    return {"mid-interval": mid, "interval-end": on_grid}


@pytest.mark.parametrize("where", ["mid-interval", "interval-end"])
def test_the_next_leg_of_a_run_stopped_by_its_wall_limit_is_the_uninterrupted_run(tmp_path, where):
    """Serial: leg one stops at a step boundary (off the output grid, or exactly at a frame), leg two is `parun -H`
    from its archive; steps, dofs and force rows equal the run that was never stopped, with no warning that the
    state was rebuilt."""
    stop_after = _stop_steps(tmp_path)[where]
    _, first, _ = _legs(tmp_path, stop_after)
    _same_run(tmp_path, 1, stop_after)
    # the stop frame carries the state: the first leg's log says so
    assert "Wall limit: checkpoint at t=" in first.log and "solver state included" in first.log


@pytest.mark.skipif(_mpiexec() is None, reason="needs mpiexec")
@pytest.mark.parametrize("where", ["mid-interval", "interval-end"])
def test_the_next_leg_on_two_ranks_is_the_uninterrupted_run(tmp_path, where):
    """The same on 2 ranks (the partition pinned, as the other parallel hot-start tests do): both ranks stop at the
    same step, and the hot start on the same ranks is the uninterrupted run to the last bit."""
    stop_after = _stop_steps(tmp_path)[where]
    _legs(tmp_path, stop_after, ranks=2)
    _same_run(tmp_path, 2, stop_after)


def test_the_next_leg_is_exact_however_slow_the_archive_is(tmp_path):
    """Every archive write takes 0.2 s longer (a slow machine: the measured checkpoint cost then passes the clock's half second of slack, as it did on the hosted runners): the stop, driven by the test clock, is still
    after the same step, and the continuation is still the uninterrupted run. Before the helper's clock fixed
    the checkpoint cost, the measured cost (frame time x state/dof bytes: 216 s here) passed the half step of slack
    and the run stopped at step 2 of 7, so the next leg repeated steps (hosted runners, proteustoolkit/proteus#1356)."""
    stop_after = _stop_steps(tmp_path)["mid-interval"]
    _legs(tmp_path, stop_after, SLOWARCHIVE=0.2)
    _same_run(tmp_path, 1, stop_after)


def test_a_checkpoint_that_costs_more_than_the_slack_stops_a_step_earlier(tmp_path):
    """The margin is the longest step + the checkpoint's cost. Steps are one clock second each and the limit is
    5.5 s: at no cost the run stops after 5 steps (5 + 1 > 5.5); with a checkpoint costing 1.5 s it stops after 4
    (3 + 1 + 1.5 = 5.5 fits, 4 + 1 + 1.5 = 6.5 does not)."""
    free = _run(tmp_path / "free", FT=FT, DTOUT=DTOUT, WALLSTEPS=5)
    costly = _run(tmp_path / "costly", FT=FT, DTOUT=DTOUT, WALLSTEPS=5, WALLCOST=1.5)
    assert free.returncode == costly.returncode == 75, costly.stderr[-1500:]
    assert len(_steps(tmp_path / "free")) == 5 and len(_steps(tmp_path / "costly")) == 4
