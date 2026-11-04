"""A hot-started run leaves its history and gauge files as the uninterrupted run would have written them.

Every file keeps its header and the rows up to the restart instant, drops the rows after it (they belong to the run
that was abandoned), and the continuation appends (proteus.Gauges.open_history_file and resume_history_files): the
point, line-integral and sampled gauges of proteus.Gauges and the legacy RANS3PF histories (timeHistory.txt,
forceHistory_p/_v.txt, momentHistory.txt, wettedAreaHistory.txt). A sampled gauge keeps the sample grid of the run it
continues.

Measured on the RANS3PF cylinder of test/HotStart_3P (14 steps, hot start at step 8, the driver of
test_hotstart_rans3pf.py): all eight files byte-identical to the uninterrupted run's. On main 9fca0b95 the files were
reopened "w": timeHistory.txt kept 6 of 15 rows, the every-step gauge 8 of 17, and the sampled gauge restarted its
grid at the restart instant.
"""
import filecmp
import io
import os
import subprocess
import sys

import pytest

from proteus import Gauges

here = os.path.dirname(os.path.abspath(__file__))
driver = os.path.join(here, "hotstart_measure", "run_rans3p.py")
STEPS, STOP = 14, 8
FILES = ("timeHistory.txt", "forceHistory_p.txt", "forceHistory_v.txt", "momentHistory.txt", "wettedAreaHistory.txt",
         "gauge_every_step.csv", "gauge_sampled.csv", "gauge_line.csv", "gauge_boundary_integrals.csv")


def _run(outdir, mode):
    env = dict(os.environ, HS_NSTEPS=str(STEPS), HS_GAUGES="1", MPLBACKEND="AGG", FI_PROVIDER="tcp",
               OMP_NUM_THREADS="1")
    args = [sys.executable, driver, str(outdir), mode] + ([str(STOP)] if mode != "full" else [])
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def _rows(path):
    with open(path) as f:
        return len(f.readlines())


def test_hot_started_history_files_are_the_uninterrupted_runs(tmp_path):
    _run(tmp_path / "full", "full")
    _run(tmp_path / "hot", "stop")
    stopped = {name: _rows(os.path.join(str(tmp_path / "hot"), name)) for name in FILES}
    _run(tmp_path / "hot", "hot")
    for name in FILES:
        full, hot = os.path.join(str(tmp_path / "full"), name), os.path.join(str(tmp_path / "hot"), name)
        assert _rows(hot) > stopped[name], name  # the continuation appended
        assert filecmp.cmp(full, hot, shallow=False), name


def _file(text):
    f = io.StringIO()
    f.write(text)
    return f


def test_keep_rows_up_to_keeps_the_header_and_the_rows_at_or_before_t():
    f = _file("time, u\n0.0, 1\n0.1, 2\n0.2, 3\n0.30000000000000004, 4\n")
    assert Gauges.keep_rows_up_to(f, 0.2) == [0.0, 0.1, 0.2]
    assert f.getvalue() == "time, u\n0.0, 1\n0.1, 2\n0.2, 3\n"


def test_keep_steps_up_to_cuts_every_member_by_its_rows_per_step():
    times = _file("0.1\n0.2\n0.3\n")
    perFlag = _file("a1\na2\nb1\nb2\nc1\nc2\n")
    perStep = _file("a\nb\nc\n")
    assert Gauges.keep_steps_up_to(times, [(perFlag, 2), (perStep, 1)], 0.2) == 2
    assert (times.getvalue(), perFlag.getvalue(), perStep.getvalue()) == ("0.1\n0.2\n", "a1\na2\nb1\nb2\n", "a\nb\n")


@pytest.fixture
def history(tmp_path):
    yield tmp_path
    Gauges.reset_history_files(False)


def test_open_history_file_keeps_a_file_only_on_a_hot_start(history):
    path = str(history / "h.txt")
    with open(path, "w") as f:
        f.write("old\n")
    Gauges.reset_history_files(False)
    f, resumed = Gauges.open_history_file(path)
    f.close()
    assert not resumed and open(path).read() == ""
    with open(path, "w") as f:
        f.write("old\n")
    Gauges.reset_history_files(True)
    f, resumed = Gauges.open_history_file(path)
    f.write("new\n")
    f.close()
    assert resumed and open(path).read() == "old\nnew\n"
