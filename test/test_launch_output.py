"""launch_output: a failed helper launch keeps its whole output (the partition-independence tests kept the last 2000
characters of each stream, which lost a 2-rank segfault's backtrace, and nothing at all of a hang)."""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("launch_output", os.path.join(HERE, "launch_output.py"))
launch_output = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch_output)

LONG = "frame %06d\n"  # 13 characters a line: 20000 lines is 260000 characters, far past any tail


def test_a_crash_keeps_its_whole_output_and_names_the_signal(tmp_path):
    code = ("import os, signal, sys\n"
            "sys.stderr.write(''.join(%r %% i for i in range(20000)))\nsys.stderr.flush()\n"
            "print('stdout line', flush=True)\nos.kill(os.getpid(), signal.SIGSEGV)\n" % LONG)
    with pytest.raises(AssertionError) as e:
        launch_output.run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), timeout=60)
    message = str(e.value)
    assert "SIGSEGV" in message and "frame 019999" in message and "stdout line" in message
    assert (tmp_path / "launch.stderr.txt").read_text().startswith("frame 000000\n")
    assert "frame 019999\n" in (tmp_path / "launch.stderr.txt").read_text()  # then faulthandler's traceback, if on
    assert str(tmp_path / "launch.stderr.txt") in message


def test_a_hang_keeps_what_it_printed(tmp_path):
    code = "import time\nprint('started the solve', flush=True)\ntime.sleep(60)\n"
    with pytest.raises(AssertionError) as e:
        launch_output.run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), timeout=3)
    assert "timed out after 3 s" in str(e.value) and "started the solve" in str(e.value)
    assert "started the solve" in (tmp_path / "launch.stdout.txt").read_text()


def test_a_success_returns_the_completed_process(tmp_path):
    out = launch_output.run([sys.executable, "-c", "print('ok')"], cwd=tmp_path, env=dict(os.environ), timeout=60)
    assert out.stdout == "ok\n" and not (tmp_path / "launch.stdout.txt").exists()


@pytest.mark.parametrize("code, status", [(1, "exit status 1"), (139, "exit status 139 (SIGSEGV?)"),
                                          (-11, "killed by SIGSEGV")])
def test_status(code, status):
    assert launch_output._status(code) == status
