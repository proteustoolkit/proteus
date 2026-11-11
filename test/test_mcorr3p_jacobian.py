"""The MCorr3P Jacobian against finite differences of its residual (see levelset/vortex2D/test_mcorr_jacobian.py for the method).

MCorr3P assembles the consistent element Jacobian by default (Coefficients.fastAssembly=False), which is exact.  With
fastAssembly=True it uses a LUMPED interface mass matrix (precomputed in the residual) plus a stiffness matrix: cheap and an M-matrix,
but only an approximation of dR/du, so Newton converges linearly.  Run on the 2D rising-bubble deck in a subprocess
(the deck is the one test_step_retry.py uses).
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "mcorr3p_jacobian_helper.py")


def _fd_errors(cwd, fast_assembly):
    os.makedirs(str(cwd), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", FASTASSEMBLY="1" if fast_assembly else "0")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    out = subprocess.run([sys.executable, HELPER], capture_output=True, text=True, timeout=1800, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    lines = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    header = [d for d in lines if "ndof" in d][0]
    return header, {d["h"]: d["rel_err"] for d in lines if "rel_err" in d}


def test_consistent_assembly_gives_the_exact_jacobian(tmp_path):
    header, err = _fd_errors(tmp_path, fast_assembly=False)
    assert header["fastAssembly"] is False and header["epsFactHeaviside"] == header["epsFactDirac"]
    assert err[1e-6] < 1e-8
    assert err[1e-6] < err[1e-4]        # and it falls with h: a consistent derivative, not a coincidence


def test_lumped_option_is_a_bounded_approximation_of_the_jacobian(tmp_path):
    header, err = _fd_errors(tmp_path, fast_assembly=True)
    assert header["fastAssembly"] is True
    # lumping the Dirac mass changes J at the 1e-3 level on this deck (3.3e-3); it must stay an approximation, not drift away
    assert err[1e-6] < 5e-2
    # and it does not fall with h: the lumping error is the same at every perturbation size
    assert err[1e-6] == pytest.approx(err[1e-4], rel=1e-2)


def test_consistent_assembly_is_the_default(tmp_path):
    import proteus.mprans.MCorr3P as M
    assert M.Coefficients.__init__.__defaults__[-1] is False
