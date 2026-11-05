"""parun's exit status reports whether the simulation finished.

A run that gives up after a failed system step ("System Step Failed") must
exit nonzero, on every rank, so batch scripts and mpiexec see the failure.
"""
import os
import shutil
import subprocess
import sys

_scriptdir = os.path.dirname(os.path.abspath(__file__))
_parun = os.path.join(_scriptdir, os.pardir, os.pardir, "scripts", "parun")


def run_parun(tmp_path, n_module):
    # parun loads the decks from the working directory and writes logs there,
    # so run on a copy rather than in the source tree
    for name in ("heat_1d_p.py", "heat_1d_n.py", "heat_1d_fail_n.py"):
        shutil.copy(os.path.join(_scriptdir, name), tmp_path)
    parun = _parun if os.path.isfile(_parun) else shutil.which("parun")
    cmd = [sys.executable, parun, "-l", "5", "-v", "-D", str(tmp_path / "data"),
           "heat_1d_p.py", n_module + ".py"]
    return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600)


def test_success_exits_zero(tmp_path):
    result = run_parun(tmp_path, "heat_1d_n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "System Step Failed" not in result.stdout


def test_system_step_failure_exits_nonzero(tmp_path):
    result = run_parun(tmp_path, "heat_1d_fail_n")
    assert "System Step Failed" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
