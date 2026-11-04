"""The conftest retries an mpiexec launch that never started, and only that.

PMIx sometimes cannot find a network interface for a moment on a CI runner, and every launch
in that second dies with "ptl_tool: problems getting address ..." before any rank runs.
"""
import subprocess
import sys

SCRIPT = """
import os, sys
path = sys.argv[1]
count = int(open(path).read()) if os.path.exists(path) else 0
open(path, 'w').write(str(count + 1))
if count < int(sys.argv[2]):
    sys.stderr.write(sys.argv[3])
    sys.exit(213)
"""


def _run(tmp_path, failures, message):
    counter = tmp_path / 'count'
    result = subprocess.run([sys.executable, '-c', SCRIPT, str(counter), str(failures), message],
                            capture_output=True, text=True, timeout=120)
    return result, int(counter.read_text())


def test_a_launch_that_never_started_is_run_again(tmp_path):
    result, runs = _run(tmp_path, 1, 'ptl_tool: problems getting address for index 0 (kernel index -1)')
    assert result.returncode == 0
    assert runs == 2


def test_any_other_failure_is_left_alone(tmp_path):
    result, runs = _run(tmp_path, 1, 'a real error')
    assert result.returncode == 213
    assert runs == 1


def test_it_gives_up_after_three_launches(tmp_path):
    result, runs = _run(tmp_path, 10, 'ptl_tool: problems getting address for index 0 (kernel index -1)')
    assert result.returncode == 213
    assert runs == 3
