"""The wall-clock budget (proteus.WallLimit): parsing, the scheduler's end time, and the decision to stop."""
import subprocess

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


def test_pbs_end_time_is_the_requested_walltime_less_the_used():
    seen = []
    deadline = WallLimit.pbs_deadline({"PBS_JOBID": "4242.pbs01"}, _run(QSTAT, seen=seen), lambda: 100.0)
    assert deadline == 100.0 + (24 - 1.5) * 3600 and seen == [["qstat", "-f", "4242.pbs01"]]
    assert WallLimit.pbs_deadline({}, _no_run, lambda: 0.0) is None
    assert WallLimit.pbs_deadline({"PBS_JOBID": "1"}, _run("Job Id: 1\n"), lambda: 0.0) is None


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
