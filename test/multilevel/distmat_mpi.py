"""Helper run under mpiexec by test_without_petsc4py.py; not collected.

Builds the same random row-distributed rectangular matrix as a petsc4py Mat and
as a LinearAlgebraTools.DistMat, and compares mult and multTranspose on random
vectors, plus a ghosted ParVec_mpi4py against ParVec_petsc4py for the forward
insert and reverse add. Needs petsc4py (it is the reference). Prints the
largest differences, and for the two products a bound on what rounding alone
can make them: k*eps*max_i sum_j |a_ij||x_j|, k the most terms in one sum. Two
evaluations of a k-term sum each lie within (k/2)*eps*sum|a||x| of the exact one
whatever their order or whether a multiply and an add are fused, which a PETSc
built by GCC for aarch64 does (GCC contracts a*b+c into an FMA by default in GNU C
mode; spark2's pip PETSc is built with -g -O). On x86-64 without -mfma it cannot.
The vector scatters only copy and add exactly representable values, so they
must agree exactly.

Usage: distmat_mpi.py NROWS NCOLS
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from petsc4py import PETSc
from proteus import LinearAlgebraTools as LAT

mpi = comm.comm.tompi4py()
rank, size = comm.rank(), comm.size()
nRows, nCols = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(12345)
# uneven layouts, some ranks possibly empty
rowCut = np.sort(rng.integers(0, nRows + 1, size - 1))
colCut = np.sort(rng.integers(0, nCols + 1, size - 1))
rowOff = np.concatenate([[0], rowCut, [nRows]])
colOff = np.concatenate([[0], colCut, [nCols]])
n, m = int(rowOff[rank + 1] - rowOff[rank]), int(colOff[rank + 1] - colOff[rank])
entries = [(r, rng.choice(nCols, size=min(nCols, 4), replace=False), rng.standard_normal(min(nCols, 4)))
           for r in range(nRows)]
sizes = [[n, nRows], [m, nCols]]
pm = PETSc.Mat().create(mpi)
pm.setType('aij')
pm.setSizes(sizes)
pm.setUp()
dm = LAT.DistMat(mpi, sizes)
for (r, cols, vals) in entries[rowOff[rank]:rowOff[rank + 1]]:
    pm.setValues([r], cols.tolist(), vals.tolist())
    dm.setValues([r], cols.tolist(), vals.tolist())
pm.assemblyBegin(); pm.assemblyEnd()
dm.assemblyBegin(); dm.assemblyEnd()

xAll = rng.standard_normal(nCols)
yAll = rng.standard_normal(nRows)
x = xAll[colOff[rank]:colOff[rank + 1]]
y = yAll[rowOff[rank]:rowOff[rank + 1]]
# what rounding alone can make two evaluations of each product differ by (every rank has every entry)
absA = np.zeros((nRows, nCols))
for (r, cols, vals) in entries:
    absA[r, cols] = np.abs(vals)
eps = np.finfo(float).eps
multBound = eps * int((absA > 0).sum(axis=1).max()) * float((absA @ np.abs(xAll)).max())
transBound = eps * int((absA > 0).sum(axis=0).max()) * float((absA.T @ np.abs(yAll)).max())
px, py = pm.createVecRight(), pm.createVecLeft()
dx, dy = dm.createVecRight(), dm.createVecLeft()
px.getArray()[:] = x; dx.getArray()[:] = x
pm.mult(px, py); dm.mult(dx, dy)
multErr = mpi.allreduce(float(np.max(np.abs(py.getArray() - dy.getArray()), initial=0.0)), op=max)
py.getArray()[:] = y; dy.getArray()[:] = y
pm.multTranspose(py, px); dm.multTranspose(dy, dx)
transErr = mpi.allreduce(float(np.max(np.abs(px.getArray() - dx.getArray()), initial=0.0)), op=max)

# ghosted vectors: each rank ghosts a random set of entries it does not own
N = nRows
owned = np.arange(rowOff[rank], rowOff[rank + 1])
others = np.setdiff1d(np.arange(N), owned)
ghosts = np.sort(rng.choice(others, size=min(others.size, 5), replace=False)) if others.size else np.array([], 'i')
s2g = np.concatenate([owned, ghosts]).astype('i')
values = np.arange(N, dtype='d') * 1.5 + 1.0
arrP = np.concatenate([values[owned], np.full(ghosts.size, -7.0)])
arrM = arrP.copy()
vp = LAT.ParVec_petsc4py(arrP, 1, n, N, ghosts.size, s2g)
vm = LAT.ParVec_mpi4py(arrM, 1, n, N, ghosts.size, s2g)
vp.scatter_forward_insert(); vm.scatter_forward_insert()
fwdErr = mpi.allreduce(float(np.max(np.abs(arrP - arrM), initial=0.0)), op=max)
fwdTruth = mpi.allreduce(float(np.max(np.abs(arrM - values[s2g]), initial=0.0)), op=max)
arrP[n:] = 2.0; arrM[n:] = 2.0
vp.scatter_reverse_add(); vm.scatter_reverse_add()
revErr = mpi.allreduce(float(np.max(np.abs(arrP[:n] - arrM[:n]), initial=0.0)), op=max)
if rank == 0:
    print("RESULT mult=%.3e multTranspose=%.3e forward=%.3e forward_truth=%.3e reverse=%.3e"
          " mult_bound=%.3e multTranspose_bound=%.3e"
          % (multErr, transErr, fwdErr, fwdTruth, revErr, multBound, transBound))
    sys.stdout.flush()
