"""A wall-clock budget for a run: when to stop at a step boundary and checkpoint (``parun --wall-limit``).

A batch job has a wall time (24 h on the HPCMP machines) after which the scheduler kills it, in the middle of a step or
of an archive write. A run given a limit instead stops by itself at a system step boundary, with room left to write
one frame and the solver state, and exits with ``CHECKPOINT_EXIT_STATUS`` so that the job script can start the next
leg (``parun -H``). Nothing here relies on the scheduler's signals: the decision is the run's own, made by rank 0 at
each step boundary and broadcast, so every rank stops at the same step.

The decision is ``used + margin > limit`` with ``margin = longest system step so far + what a checkpoint costs to
write`` (the frames and the solver state of every model, as measured by the run's own writes).
"""
import os
import re
import subprocess
import time

#: parun's exit status when it stopped at a step boundary for its wall limit, with a checkpoint written: sysexits'
#: EX_TEMPFAIL ("temporary failure; the user is invited to retry"). 0 is a finished run, 1 a failed one.
CHECKPOINT_EXIT_STATUS = 75

_DURATION = re.compile(r"^(?:(\d+)-)?(\d+):(\d{1,2})(?::(\d{1,2}))?$")


def parse_duration(text):
    """Seconds in ``HH:MM:SS`` (hours may exceed 24), ``D-HH:MM:SS`` (SLURM's long form) or ``MM:SS``.

    Raises ``ValueError`` for anything else, so a typo in a job script stops the run at the start rather than
    running without the limit it was meant to have."""
    m = _DURATION.match(str(text).strip())
    if m is None:
        raise ValueError("a wall limit is HH:MM:SS (or D-HH:MM:SS), not %r" % (text,))
    days, first, second, third = m.groups()
    if third is None:
        hours, minutes, seconds = 0, int(first), int(second)
    else:
        hours, minutes, seconds = int(first), int(second), int(third)
    if minutes > 59 or seconds > 59:
        raise ValueError("a wall limit is HH:MM:SS (minutes and seconds below 60), not %r" % (text,))
    return ((int(days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds


def _query(command, run):
    """A scheduler command's stdout, or None when it is not there, fails or hangs."""
    try:
        done = run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def slurm_deadline(environ=None, run=subprocess.run, now=time.time):
    """The epoch time the SLURM job ends, or None (not in a SLURM job, or the end is not known).

    ``SLURM_JOB_END_TIME`` when the environment has it, else the time left that ``squeue -o %L`` reports."""
    environ = os.environ if environ is None else environ
    end = environ.get("SLURM_JOB_END_TIME")
    if end:
        try:
            return float(end)
        except ValueError:
            pass
    job = environ.get("SLURM_JOB_ID")
    if not job:
        return None
    out = _query(["squeue", "-h", "-j", job, "-o", "%L"], run)
    left = (out or "").strip().splitlines()
    if not left:
        return None
    try:
        return now() + parse_duration(left[0].strip())
    except ValueError:  # UNLIMITED, INVALID, NOT_SET
        return None


def _pbs_start(out, now):
    """The epoch time ``qstat -f`` says the job started (``stime``, PBS Pro; ``start_time``, Torque), read as local
    time (the server's and the node's clocks and zone agree on a cluster), or None when it is absent, unreadable
    or implausible (in the future, or over a year ago)."""
    m = re.search(r"^\s*(?:stime|start_time)\s*=\s*(.+?)\s*$", out, re.MULTILINE)
    if m is None:
        return None
    try:
        start = time.mktime(time.strptime(" ".join(m.group(1).split()), "%a %b %d %H:%M:%S %Y"))
    except (ValueError, OverflowError):
        return None
    current = now()
    return start if current - 366 * 86400 < start <= current + 60 else None


def pbs_deadline(environ=None, run=subprocess.run, now=time.time):
    """The epoch time the PBS job ends, or None: the job's start (``stime``) plus its ``Resource_List.walltime``,
    both from ``qstat -f`` (the same fields in PBS Pro and Torque).

    Not the walltime less ``resources_used.walltime``: PBS refreshes the used walltime only every so often (it read
    00:00:05 at 14-22 s into a job on Carpenter), which put the end 12-17 s too late. That difference is the
    fallback only, for a ``qstat`` that lists no usable start time."""
    environ = os.environ if environ is None else environ
    job = environ.get("PBS_JOBID")
    if not job:
        return None
    out = _query(["qstat", "-f", job], run)
    if not out:
        return None
    fields = dict(re.findall(r"^\s*([\w.]+)\s*=\s*(\S+)\s*$", out, re.MULTILINE))
    try:
        walltime = parse_duration(fields["Resource_List.walltime"])
    except (KeyError, ValueError):
        return None
    start = _pbs_start(out, now)
    if start is not None:
        return start + walltime
    try:
        return now() + walltime - parse_duration(fields["resources_used.walltime"])
    except (KeyError, ValueError):
        return None


def scheduler_deadline(environ=None, run=subprocess.run, now=time.time):
    """The epoch time the batch job this process runs in ends (SLURM, else PBS), or None outside a job."""
    for find in (slurm_deadline, pbs_deadline):
        deadline = find(environ, run, now)
        if deadline is not None:
            return deadline
    return None


def resolve_deadline(limit, start, environ=None, run=subprocess.run, now=time.time):
    """The epoch time a run must have stopped by, and where it came from, ``(deadline, source)``; ``(None, None)``
    when there is no limit.

    ``limit`` is parun's ``--wall-limit``: None (no limit), ``"HH:MM:SS"`` (from ``start``, when the run began), or
    ``"scheduler"`` (the batch job's end time). A duration inside a batch job with a known end is cut at the end,
    whichever comes first: a limit longer than the job's would never be reached."""
    if limit is None:
        return None, None
    if str(limit).strip().lower() == "scheduler":
        deadline = scheduler_deadline(environ, run, now)
        return (deadline, "the scheduler's end time") if deadline is not None else (None, None)
    deadline, source = start + parse_duration(limit), "--wall-limit %s" % (limit,)
    scheduled = scheduler_deadline(environ, run, now)
    if scheduled is not None and scheduled < deadline:
        deadline, source = scheduled, "the scheduler's end time (before --wall-limit %s)" % (limit,)
    return deadline, source


class WallClock:
    """The budget of one run and what it has measured of itself; ``due()`` is the decision.

    ``clock`` returns the epoch time (``time.time``, or a fake in a test). ``stepBoundary()`` is called at each
    system step boundary, ``archiving(seconds)`` for each archive write, and ``checkpointCost`` is what one
    checkpoint is measured to take."""

    def __init__(self, deadline, source="", clock=time.time):
        self.deadline = deadline
        self.source = source
        self.clock = clock
        self.longestStep = 0.0
        self.checkpointCost = 0.0
        self._archiving = 0.0
        self._boundary = None
        self._archivingAtBoundary = 0.0

    def archiving(self, seconds):
        """An archive write of ``seconds`` has happened since the last boundary: it is not part of the step."""
        self._archiving += seconds

    def stepBoundary(self):
        """A system step has just ended: its duration (without the archive writes in it) may be the longest."""
        now = self.clock()
        if self._boundary is not None:
            self.longestStep = max(self.longestStep, now - self._boundary - (self._archiving - self._archivingAtBoundary))
        self._boundary = now
        self._archivingAtBoundary = self._archiving
        return now

    def margin(self):
        """What the run must still be able to do after the step it is about to take: that step, at the longest
        one so far, and the checkpoint."""
        return self.longestStep + self.checkpointCost

    def due(self):
        """True when the next step plus a checkpoint no longer fits before the deadline. Called at a step
        boundary, on rank 0 only: the other ranks follow its answer."""
        return self.deadline is not None and self.clock() + self.margin() > self.deadline

    def left(self):
        return None if self.deadline is None else self.deadline - self.clock()
