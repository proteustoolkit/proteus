"""parun takes the case as a positional argument or as -f, with and without --TwoPhaseFlow.

The air-water-vv READMEs (and older docs) run `mpiexec -n 2 parun -f dambreak.py --TwoPhaseFlow`. The
--TwoPhaseFlow branch read the case from args[0] and died with IndexError: list index out of range on
every rank when the case was given only with -f. Measured (ceds-model, proteus b9d98f5d):

    before  parun -f case.py --TwoPhaseFlow          IndexError: list index out of range
    after   the case is imported (the test's case raises a marker error on import)
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PARUN = os.path.join(HERE, os.pardir, "scripts", "parun")
PARUN = PARUN if os.path.isfile(PARUN) else shutil.which("parun")


def _parun(tmp_path, *args):
    (tmp_path / "marker_case.py").write_text('raise RuntimeError("CASE_WAS_IMPORTED")\n')
    env = dict(os.environ, OMP_NUM_THREADS="1", FI_PROVIDER=os.environ.get("FI_PROVIDER", "tcp"))
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    out = subprocess.run([sys.executable, PARUN, *args], capture_output=True, text=True, timeout=120,
                         env=env, cwd=str(tmp_path))
    return out.returncode, out.stdout + out.stderr


@pytest.mark.parametrize("args", [
    ("marker_case.py", "--TwoPhaseFlow"),
    ("-f", "marker_case.py", "--TwoPhaseFlow"),
    ("--TwoPhaseFlow", "-f", "marker_case.py"),
])
def test_the_case_is_imported(tmp_path, args):
    code, text = _parun(tmp_path, *args)
    assert code != 0
    assert "CASE_WAS_IMPORTED" in text, text[-2000:]
    assert "IndexError" not in text


@pytest.mark.parametrize("log_level", ["1", "0"])
def test_no_case_says_how_to_give_one(tmp_path, log_level):
    code, text = _parun(tmp_path, "--TwoPhaseFlow", "-l", log_level)
    assert code != 0
    assert "Pass setup problem as -f fileName.py" in text or "No input file specified" in text, text[-2000:]
    assert "IndexError" not in text
