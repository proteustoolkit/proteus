"""The wall-clock budget (proteus.WallLimit): parsing, the scheduler's end time, and the decision to stop."""
import subprocess
import time

import pytest

from proteus import WallLimit
from proteus.WallLimit import WallClock, parse_duration, resolve_deadline


@pytest.mark.parametrize("text, seconds", [("23:50:00", 85800), ("00:00:30", 30), ("100:00:00", 360000),
                                           ("1-00:00:00", 86400), ("2-01:02:03", 176523), ("10:05", 605)])
def test_durations(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "24", "12:60:00", "1:2:3:4", "-5", "UNLIMITED", "00:00:00x"])
def test_a_malformed_limit_is_an_error(text):
    with pytest.raises(ValueError, match="wall limit"):
        parse_duration(text)


def _run(stdout="", returncode=0, seen=None):
    def run(command, **kwargs):
        if seen is not None:
            seen.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout, "")
    return run


def _no_run(command, **kwargs):
    raise AssertionError("ran %r" % (command,))


def test_slurm_end_time_from_the_environment_or_squeue():
    assert WallLimit.slurm_deadline({"SLURM_JOB_END_TIME": "1700000000"}, _no_run, lambda: 5.0) == 1700000000.0
    seen = []
    deadline = WallLimit.slurm_deadline({"SLURM_JOB_ID": "77"}, _run("1-02:03:04\n", seen=seen), lambda: 1000.0)
    assert deadline == 1000.0 + 93784 and seen == [["squeue", "-h", "-j", "77", "-o", "%L"]]
    # no job, an unlimited one, a failing squeue
    assert WallLimit.slurm_deadline({}, _no_run, lambda: 0.0) is None
    assert WallLimit.slurm_deadline({"SLURM_JOB_ID": "77"}, _run("UNLIMITED\n"), lambda: 0.0) is None
    assert WallLimit.slurm_deadline({"SLURM_JOB_ID": "77"}, _run("", returncode=1), lambda: 0.0) is None

    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])
    assert WallLimit.slurm_deadline({"SLURM_JOB_ID": "77"}, missing, lambda: 0.0) is None


QSTAT = """Job Id: 4242.pbs01
    Job_Name = flume
    resources_used.walltime = 01:30:00
    Resource_List.walltime = 24:00:00
    Resource_List.select = 32:ncpus=48
"""
#a job that started at 01:02:03 and has used 00:00:05 of its walltime according to PBS, which refreshes the figure only
#every so often: the figure read 12-17 s short of the time since the start on Carpenter
STIME = "Fri Nov 13 01:02:03 2026"
QSTAT_STIME = QSTAT.replace("01:30:00", "00:00:05") + "    stime = %s\n" % STIME


def _epoch(text):
    return time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y"))


def test_pbs_end_time_is_the_start_plus_the_requested_walltime():
    start = _epoch(STIME)
    now = start + 19.0  # parun is 19 s into the job, qstat says 5 s were used
    seen = []
    deadline = WallLimit.pbs_deadline({"PBS_JOBID": "4242.pbs01"}, _run(QSTAT_STIME, seen=seen), lambda: now)
    assert deadline == start + 24 * 3600 and seen == [["qstat", "-f", "4242.pbs01"]]
    # the stale used-walltime rule would have put the end 14 s late
    assert now + (24 - 5 / 3600.0) * 3600 - deadline == pytest.approx(14.0)
    assert WallLimit.pbs_deadline({}, _no_run, lambda: 0.0) is None
    assert WallLimit.pbs_deadline({"PBS_JOBID": "1"}, _run("Job Id: 1\n"), lambda: 0.0) is None


@pytest.mark.parametrize("stime", ["Thu Nov  5 01:02:03 2026", "Thu Nov 5 01:02:03 2026"])
def test_pbs_start_time_with_a_padded_or_plain_day(stime):
    start = _epoch("Thu Nov 5 01:02:03 2026")
    out = QSTAT.replace("01:30:00", "00:10:00") + "    stime = %s\n" % stime
    assert WallLimit.pbs_deadline({"PBS_JOBID": "7"}, _run(out), lambda: start + 600.0) == start + 24 * 3600


def test_pbs_torque_start_time_field():
    start = _epoch(STIME)
    out = QSTAT + "    start_time = %s\n" % STIME
    assert WallLimit.pbs_deadline({"PBS_JOBID": "7"}, _run(out), lambda: start + 30.0) == start + 24 * 3600


@pytest.mark.parametrize("stime", ["", "    stime = not a date\n", "    stime = Fri Nov 13 01:02:03 2036\n",
                                   "    stime = Fri Nov 13 01:02:03 2020\n"])
def test_pbs_falls_back_to_the_used_walltime_without_a_plausible_start(stime):
    now = _epoch(STIME) + 100.0
    deadline = WallLimit.pbs_deadline({"PBS_JOBID": "4242.pbs01"}, _run(QSTAT + stime), lambda: now)
    assert deadline == now + (24 - 1.5) * 3600


def test_the_limit_is_cut_at_the_schedulers_end_and_a_typo_is_not_swallowed():
    env = {"SLURM_JOB_END_TIME": "1000"}
    # a limit past the job's end is cut at the job's end
    assert resolve_deadline("24:00:00", 0.0, env, _no_run) == (1000.0, "the scheduler's end time (before --wall-limit 24:00:00)")
    # a limit before it stands
    assert resolve_deadline("00:10:00", 0.0, env, _no_run) == (600.0, "--wall-limit 00:10:00")
    assert resolve_deadline("scheduler", 0.0, env, _no_run) == (1000.0, "the scheduler's end time")
    # outside a job: the duration alone; "scheduler" has nothing to read
    assert resolve_deadline("00:10:00", 50.0, {}, _no_run) == (650.0, "--wall-limit 00:10:00")
    assert resolve_deadline("scheduler", 0.0, {}, _no_run) == (None, None)
    assert resolve_deadline(None, 0.0, env, _no_run) == (None, None)
    with pytest.raises(ValueError):
        resolve_deadline("23-50-00", 0.0, {}, _no_run)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_decision_keeps_the_longest_step_and_a_checkpoint_in_hand():
    clock = Clock()
    wc = WallClock(1000.0, clock=clock)
    wc.checkpointCost = 30.0
    wc.stepBoundary()
    assert not wc.due()                    # nothing measured: 0 + 30 fits
    for duration in (10.0, 40.0, 20.0):    # the longest step is 40 s
        clock.t += duration
        wc.stepBoundary()
    assert wc.longestStep == 40.0 and wc.margin() == 70.0
    clock.t = 929.0
    assert not wc.due() and wc.left() == pytest.approx(71.0)
    clock.t = 931.0                         # 931 + 70 > 1000: no room for another step and the checkpoint
    assert wc.due()


def test_archive_writes_are_not_part_of_a_step():
    clock = Clock()
    wc = WallClock(1000.0, clock=clock)
    wc.stepBoundary()
    clock.t += 50.0
    wc.archiving(45.0)                      # 45 of the 50 s were an archive write
    wc.stepBoundary()
    assert wc.longestStep == pytest.approx(5.0)


def test_no_deadline_is_never_due():
    wc = WallClock(None, clock=Clock())
    wc.stepBoundary()
    assert not wc.due() and wc.left() is None


def test_the_stop_marker_is_written_whole_read_back_and_cleared(tmp_path):
    path = WallLimit.marker_path("flume", str(tmp_path))
    assert path.endswith("flume.checkpoint") and WallLimit.read_marker(path) is None
    WallLimit.write_marker(path, 0.0030000000000000005, 7, 1536, now=lambda: 1700000000.4)
    assert WallLimit.read_marker(path) == {"t": 0.0030000000000000005, "frame": 7, "ranks": 1536,
                                           "exit_status": 75, "written": 1700000000}
    assert [p.name for p in tmp_path.iterdir()] == ["flume.checkpoint"]  # no temporary file left
    WallLimit.write_marker(path, 0.5, 9, 4)  # a later stop replaces it
    assert WallLimit.read_marker(path)["frame"] == 9
    WallLimit.clear_marker(path)
    WallLimit.clear_marker(path)  # already gone: not an error
    assert WallLimit.read_marker(path) is None


class _Model:
    def __init__(self, name):
        self.name = name


def _ns(frames, ratio=20.0, state=None, read=None):
    """The part of a NumericalSolution _checkpointCost looks at: per model its frame write times, the state's size
    over the dofs' (``ratio``), the state writes timed so far, the state reads a hot start timed."""
    from types import SimpleNamespace
    from proteus.NumericalSolution import NS_base
    ns = SimpleNamespace(modelList=[_Model(n) for n in frames],
                         _frameSamples={n: list(t) for n, t in frames.items()},
                         _frameTimes={n: max(t) for n, t in frames.items()},
                         _stateTimes=state or {}, _stateReadTimes=read or {},
                         _stateToFrameRatio=lambda model: ratio,
                         _STATE_WRITE_PER_READ=NS_base._STATE_WRITE_PER_READ)
    return NS_base._checkpointCost(ns)


def test_one_slow_frame_does_not_set_the_state_estimate():
    """Torino part B: rans2p frames took 0.10-0.12 s except one of 1.57 s. The state estimate was that frame time
    times the state/dof bytes (here 20): 31 s for a state that takes ~2 s to write. The median frame is 0.11 s."""
    frames = {"rans2p": [0.11, 0.10, 1.57, 0.12, 0.11]}
    cost = _ns(frames)
    assert cost == pytest.approx(1.57 + 0.11 * 20.0)
    assert 3.0 < cost < 5.0  # against 1.57 * 21 = 33 s with the longest frame


def test_a_timed_state_write_replaces_the_estimate():
    assert _ns({"vof": [0.1, 5.0]}, state={"vof": 1.3}) == pytest.approx(5.0 + 1.3)


def test_a_hot_starts_state_read_is_the_estimate_in_a_continued_leg():
    frames = {"ncls": [0.05, 0.04, 0.9]}
    assert _ns(frames, read={"ncls": 0.6}) == pytest.approx(0.9 + 2.0 * 0.6)
    # a state write timed in this leg beats the read
    assert _ns(frames, read={"ncls": 0.6}, state={"ncls": 0.7}) == pytest.approx(0.9 + 0.7)


def test_no_frame_written_yet_costs_nothing():
    from types import SimpleNamespace
    from proteus.NumericalSolution import NS_base
    ns = SimpleNamespace(modelList=[_Model("rdls")], _frameSamples={}, _frameTimes={}, _stateTimes={},
                         _stateReadTimes={}, _STATE_WRITE_PER_READ=2.0)
    assert NS_base._checkpointCost(ns) == 0.0
