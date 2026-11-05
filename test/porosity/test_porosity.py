import subprocess
import os
import re
import pytest

@pytest.mark.porosity

def teardown_method():
    """ Tear down function """
    FileList = ['sloshing.xmf', 'sloshing.ymf',
                'sloshing.h5',
                'forceHistory_p.txt',
                'forceHistory_v.txt',
                'momentHistory.txt',
                'wettedAreaHistory.txt',
                'timeHistory.txt',
                'sloshing.log'
    ]
    for file in FileList:
        if os.path.isfile(file):
            os.remove(file)
        else:
            pass

def test_runTwoPhaseFlow_wPorosity_2D(case_copy):
    """Test restart workflow (in a copy of this directory, so the run's files stay out of the checkout)"""
    currentPath = str(case_copy(os.path.dirname(os.path.abspath(__file__))))
    runCommand = "cd "+currentPath+"; parun --TwoPhaseFlow -l5 -v -C \"T=20. he=0.025 openTop=False genMesh=False\" sloshing.py;"
    subprocess.run(runCommand,shell=True, check=True, timeout=600)
    check_massConservation_wPorosity_2D(currentPath)

def check_massConservation_wPorosity_2D(currentPath, mcorr_nl_atol_res=1e-10):
    """Reads the sloshing.log the run above writes, so it is part of that test."""
    file=open(currentPath+'/sloshing.log','r')
    lines=file.readlines()
    mcorr=[]
    nSteps=0
    for i,j in enumerate(lines):
        if "Phase 0 mass (consistent) after mass correction" in j:
            # The next line is "[<elapsed>] Step Taken, Model step t= <t>, ...".
            # Profiling writes the elapsed time as "[%8.4f]", which is no
            # longer padded with a space after "[" once the run passes 100 s,
            # so t is found by name rather than by its field number.
            t = re.search(r"\bt=\s*([-+.0-9eE]+),", lines[i+1]).group(1)
            mcorr.append([float(t),float(j.split()[-1])])
            nSteps +=1
    if (mcorr[0][1]+nSteps*mcorr_nl_atol_res)<=mcorr[-1][-1] or (mcorr[0][1]-nSteps*mcorr_nl_atol_res)>=mcorr[-1][-1]:
        pytest.fail("Mass is not being conserved within nonlinear residual tolerance for MCorr. Absolute mass error: {0:e}\n".format(abs(mcorr[-1][-1]-mcorr[0][1])))