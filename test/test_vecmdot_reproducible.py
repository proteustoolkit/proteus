"""PETSc's VecMDot is the same bits whatever the alignment of the vectors (C0 round 11).

GMRES with classical Gram-Schmidt (PETSc's default) calls VecMDot once per iteration. PETSc >= 3.21 forms it with
one BLAS gemv over the Krylov vectors (-vec_mdot_use_gemv, on by default) and OpenBLAS's gemv sums in an order that
depends on where malloc put those vectors: a CLSVOF dam break on two ranks, with nothing changed, differed in the last
bit of a linear solve in 1 run of 20 and then in every dataset after it (cekees/proteus#256). proteus.Comm.init now
turns the option off unless the user set it. The test builds the same eleven-or-more vectors under many different heap
states and asks for one answer.
"""
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("petsc4py")

SCRIPT = r"""
import hashlib, random, sys
import numpy as np
from proteus import Comm
Comm.init()
from petsc4py import PETSc
n, nv, trials = 540, 30, 400
rng = np.random.default_rng(1)
data = [rng.standard_normal(n) for _ in range(nv + 1)]
random.seed(5)
answers, keep = set(), []
for t in range(trials):
    junk = [bytearray(random.randrange(1, 300)) for _ in range(random.randrange(1, 40))]
    y = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    vs = []
    for i in range(nv):
        junk.append(bytearray(random.randrange(1, 200)))
        v = y.duplicate()
        v.setArray(data[i])
        vs.append(v)
    y.setArray(data[nv])
    answers.add(hashlib.md5(np.asarray(y.mDot(vs)).tobytes()).hexdigest())
    if t % 7 == 0:
        keep.append(junk)
print("ANSWERS", len(answers))
"""


def _answers(tmp_path, petsc_options=None):
    script = tmp_path / "mdot.py"
    script.write_text(SCRIPT)
    env = dict(os.environ)
    if petsc_options is not None:
        env["PETSC_OPTIONS"] = petsc_options
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    return int(re.search(r"ANSWERS (\d+)", out.stdout).group(1))


def test_vecmdot_gives_one_answer_whatever_the_heap_looks_like(tmp_path):
    assert _answers(tmp_path) == 1


def test_a_value_set_by_the_user_is_kept(tmp_path):
    # not asserted to vary (older PETSc, other BLAS): it must run, and it must not be overridden by proteus
    assert _answers(tmp_path, "-vec_mdot_use_gemv 1") >= 1
