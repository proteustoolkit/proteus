#!/usr/bin/env python
"""Helper for test_mcorr3p_jacobian.py: run the 2D rising-bubble deck (RANS3PF, VOF3P, NCLS3P, RDLS, MCorr3P) for a few steps through
step_retry_mpi.py, then print the finite-difference error of the MCorr3P Jacobian, one JSON line per perturbation size.
FASTASSEMBLY=0 selects the consistent element Jacobian (Coefficients.fastAssembly=False, the default), FASTASSEMBLY=1 the lumped one; unset keeps the default.  Run in an empty directory."""
import json
import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.update(DIR="surface_tension/rising_bubble_rans3p", SO="risingBubble_so.py", OPTS="nd=2 USE_SUPG_NS=1 ARTIFICIAL_VISCOSITY_NS=1",
                  FT=os.environ.get("FT", "0.004"))
import sys
sys.path.insert(0, HERE)
import fd_jacobian
import proteus.NumericalSolution as NS

_calculateSolution = NS.NS_base.calculateSolution


def _fd_check(ns):
    mc = [m.levelModelList[-1] for m in ns.modelList if type(m.levelModelList[-1]).__module__.endswith("MCorr3P")][0]
    if os.environ.get("FASTASSEMBLY") in ("0", "1"):
        mc.coefficients.fastAssembly = os.environ["FASTASSEMBLY"] == "1"
    print(json.dumps(dict(ndof=len(mc.u[0].dof), fastAssembly=bool(mc.coefficients.fastAssembly), epsFactHeaviside=mc.coefficients.epsFactHeaviside,
                          epsFactDirac=mc.coefficients.epsFactDirac)), flush=True)
    for h in (1e-4, 1e-5, 1e-6):
        print(json.dumps(dict(h=h, rel_err=float(fd_jacobian.levelmodel_error(mc, h)))), flush=True)


def calculateSolution(self, *args, **kwargs):
    out = _calculateSolution(self, *args, **kwargs)
    _fd_check(self)
    return out


NS.NS_base.calculateSolution = calculateSolution
runpy.run_path(os.path.join(HERE, "step_retry_mpi.py"), run_name="__main__")
