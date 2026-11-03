"""RANS3PF's viscous boundary force and moment do not depend on how the pressure term is written.

The kernel divided the momentum flux by dmom_u_ham_grad_p, which carries aux_pressure: 0 when the pressure is integrated
by parts or the pressure term is killed, as in every test/TwoPhaseFlow deck, so forceHistory_v.txt and momentHistory.txt
were NaN in every row there (damBreak, risingBubble, TwoDimBucklingFlow, fillingTank). It now divides by the factor the
flux itself carries, phi_s_effect * porosity, which is the old divisor whenever aux_pressure is 1.

The cylinder deck (test/cylinder2D/conforming_rans3p, aux_pressure 1 by default) is run twice: as it is, and with the
pressure integrated by parts (aux_pressure 0). With exact quadrature the two are the same discrete problem, so the
forces must agree; measured to 1.2e-14 (viscous) and 2.5e-13 (pressure) relative at t=8, and bit-identical to the
old kernel for the first run.
"""
import os
import shutil
import subprocess
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(cwd, integrate_by_parts):
    for v in ("rans3p", "rans2p"):   # the rans3p deck reads its mesh section from ../conforming_rans2p
        shutil.copytree(os.path.join(HERE, "cylinder2D", "conforming_" + v), os.path.join(str(cwd), "conforming_" + v),
                        ignore=shutil.ignore_patterns("__pycache__", "*.h5", "*.ymf"))
    deck = os.path.join(str(cwd), "conforming_rans3p")
    if integrate_by_parts:
        p = os.path.join(deck, "twp_navier_stokes_p.py")
        text = open(p).read()
        old = "coefficients = RANS3PF.Coefficients("
        assert old in text
        open(p, "w").write(text.replace(old, old + "INT_BY_PARTS_PRESSURE=1, "))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    parun = shutil.which("parun") or os.path.join(os.path.dirname(sys.executable), "parun")
    out = subprocess.run([sys.executable, parun, "-l1", "cylinder_so.py", "-C",
                          "T=0.2 he=0.03 vspaceOrder=2 onlySaveFinalSolution=True"],
                         capture_output=True, text=True, timeout=900, env=env, cwd=deck)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return {f: numpy.loadtxt(os.path.join(deck, f + ".txt")) for f in ("forceHistory_p", "forceHistory_v", "momentHistory")}


def test_viscous_force_is_finite_and_the_same_with_the_pressure_integrated_by_parts(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = _run(tmp_path / "a", integrate_by_parts=False)
    by_parts = _run(tmp_path / "b", integrate_by_parts=True)
    for name in plain:
        assert numpy.isfinite(by_parts[name]).all(), name
        scale = numpy.abs(plain[name]).max()
        assert scale > 0.0, name
        assert numpy.abs(plain[name] - by_parts[name]).max() < 1e-10 * scale, name
