#!/usr/bin/env python

import subprocess
import os
import pytest

RUN = "cd %s; parun -C \"gen_mesh={gen} usePUMI=True adapt={adapt} fixedTimeStep=False\" -D \"{name}\" dambreak_Colagrossi_so.py;"

def test_MeshAdaptRestart(verbose=0):
    """Generate the mesh, run the baseline and the adapted restart, and compare their pressure
    gauges. The runs read the mesh the first one writes and the comparison reads their output,
    so they are one test: it does not depend on another test having run, or on where one ran."""
    currentPath = os.path.dirname(os.path.abspath(__file__))
    for gen_mesh, adapt, name in (("True", 0, "baseline"),  # generate the PUMI mesh
                                  ("False", 0, "baseline"),  # baseline data
                                  ("False", 1, "adapt_0")):  # restart workflow
        subprocess.check_call(RUN.format(gen=gen_mesh, adapt=adapt, name=name) % currentPath, shell=True)
    #subprocess.call("diff normal adapt_0/pressure.csv baseline/pressure.csv")
    with open(currentPath+'/baseline/pressure.csv') as file1, open(currentPath+'/adapt_0/pressure.csv') as file2:
        for line1, line2 in zip(file1, file2):
            if(line1 != line2):
                pytest.fail("pressure gauge values are not the same!\n")
