import subprocess
import os
import shutil
import pytest

@pytest.mark.slowTest
def test_workflowPUMI(case_copy, verbose=0):
    """Test serial workflow: load model and mesh, solve within proteus
    estimate error, adapt, solve again. It's not so important if the
    problem isn't setup properly"""
    # in a copy, so the run's files stay out of the checkout; couette.py reads ../Couette.{null,msh}
    here = os.path.dirname(os.path.abspath(__file__))
    copy = case_copy(here)
    for name in ("Couette.null", "Couette.msh"):
        shutil.copy(os.path.join(here, os.pardir, name), str(copy.parent))
    currentPath = str(copy)
    runCommand = "cd "+currentPath+"; parun -l5 couette_so.py;"
    subprocess.run(runCommand,shell=True, check=True, timeout=900)
