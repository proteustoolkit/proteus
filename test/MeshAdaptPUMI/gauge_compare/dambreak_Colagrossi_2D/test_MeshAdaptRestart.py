#!/usr/bin/env python

import subprocess
import os
import pytest

RUN = "cd %s; parun -C \"gen_mesh={gen} usePUMI=True adapt={adapt} fixedTimeStep=False\" -D \"{name}\" dambreak_Colagrossi_so.py;"

#@pytest.mark.slowTest

def test_MeshAdaptRestart(case_copy, verbose=0):
    """Generate the PUMI mesh, run the baseline and the adapted restart, and compare their gauges.

    The runs read the mesh the first one writes and the comparison reads their output, so they
    are one test: it does not depend on another test having run, or on the worker that ran it.

    The point gauges (pressure) and the line integral gauges (equivalent water height) are
    rebuilt after each adapt, and must continue the baseline's output."""
    currentPath = str(case_copy(os.path.dirname(os.path.abspath(__file__))))    # run's files stay out of the checkout
    for gen_mesh, adapt, name in (("True", 0, "baseline"),  # generate the PUMI mesh
                                  ("False", 0, "baseline"),  # baseline data
                                  ("False", 1, "adapt_0")):  # restart workflow
        subprocess.run(RUN.format(gen=gen_mesh, adapt=adapt, name=name) % currentPath, shell=True, check=True,
                       timeout=900)
    #subprocess.call("diff normal adapt_0/pressure.csv baseline/pressure.csv")
    for gauge_file in ("pressure.csv", "vof_height.csv"):
        with open(currentPath+'/baseline/'+gauge_file) as file1, open(currentPath+'/adapt_0/'+gauge_file) as file2:
            rows = 0
            for line1, line2 in zip(file1, file2):
                rows += 1
                if(line1 != line2):
                    pytest.fail("%s values are not the same!\n" % gauge_file)
            assert rows > 1, "%s has no samples" % gauge_file
