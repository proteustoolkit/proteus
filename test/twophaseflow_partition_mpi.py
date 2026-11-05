#!/usr/bin/env python
"""A test/TwoPhaseFlow deck (DECK, with OPTS for its context options) to t=FT, for the partition tests.

Run under mpiexec by test_twophaseflow_partition_independence.py in an empty directory; copies the deck and the stored
meshes next to it and leaves u.csv there: one row per owned dof of every model component, sorted by
(100 * model + component, x, y, z): key, x, y, z, value, and models.csv: key, model name. ATOL=a sets every model's
nonlinear atol to a and its linear atol to a/100 (relative tolerances off); without it the deck's own tolerances (nl_atol_res ~1e-3) apply.
MAXITS=n raises every model's maxNonlinearIts to at least n (RDLS needs it at a tight atol). DIRECT=1 replaces the
serial-only LU of a deck by a PETSc direct solve (KSP preonly, LU from SuperLU_DIST), in serial and in parallel.
DIR=<directory under test/> and SO=<so file> run an so-file deck instead (every file of DIR is copied; OPTS is the
whole -C string and FT is not used). FIXNULLSPACE=1 sets PresInc's fixNullSpace.
"""
import glob, os, runpy, shutil, sys
import numpy as np

SO = os.environ.get("SO")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("DIR", "TwoPhaseFlow"))
DECK = os.environ.get("DECK", "damBreak.py")
for f in (glob.glob(os.path.join(SRC, "*")) if SO else [os.path.join(SRC, DECK)] + glob.glob(os.path.join(SRC, "mesh*"))):
    if os.path.isfile(f):
        shutil.copy(f, ".")

from proteus import Comm
from proteus import NumericalSolution as NS

def boundary_integral_gauges(pList, nList):
    """BIGAUGES=flags (comma list): a BoundaryIntegralGauges of every quantity on those flags, with a moment point on
    the first, attached to the RANS2P/RANS3PF model; it writes boundary_integrals.csv."""
    from proteus.Gauges import BoundaryIntegralGauges
    for p, n in zip(pList, nList):
        if type(p.coefficients).__module__ in ("proteus.mprans.RANS2P", "proteus.mprans.RANS3PF"):
            flags = [int(f) for f in os.environ["BIGAUGES"].split(",")]
            gauge = BoundaryIntegralGauges(tags=flags, moment_points={str(flags[0]): (0.1, 0.2, 0.0)},
                                           fileName="boundary_integrals.csv")
            n.auxiliaryVariables = list(getattr(n, "auxiliaryVariables", [])) + [gauge]


_init = NS.NS_base.__init__
def init(self, so, pList, nList, *args, **kwargs):
    if os.environ.get("BIGAUGES"):
        boundary_integral_gauges(pList, nList)
    if os.environ.get("ATOL"):
        atol = float(os.environ["ATOL"])
        for n in nList:
            n.nl_atol_res, n.l_atol_res, n.l_rtol_res, n.linTolFac, n.tolFac = atol, 0.01 * atol, 0.0, 0.0, 0.0
    if os.environ.get("DIRECT"):
        from petsc4py import PETSc
        from proteus import LinearSolvers
        for i, n in enumerate(nList):
            if n.multilevelLinearSolver is LinearSolvers.LU or n.levelLinearSolver is LinearSolvers.LU:
                n.multilevelLinearSolver = n.levelLinearSolver = LinearSolvers.KSP_petsc4py
                n.linearSmoother = None
                if n.linear_solver_options_prefix is None:
                    n.linear_solver_options_prefix = "direct%d_" % i
                for k, v in (("ksp_type", "preonly"), ("pc_type", "lu"), ("pc_factor_mat_solver_type", os.environ.get("DIRECT_PACKAGE", "superlu_dist"))):
                    PETSc.Options().setValue(n.linear_solver_options_prefix + k, v)
    if os.environ.get("FIXNULLSPACE"):
        for p in pList:
            if type(p.coefficients).__module__ == "proteus.mprans.PresInc":
                p.coefficients.fixNullSpace = True
    if os.environ.get("MAXITS"):
        for n in nList:
            n.maxNonlinearIts = max(getattr(n, "maxNonlinearIts", 0), int(os.environ["MAXITS"]))
    return _init(self, so, pList, nList, *args, **kwargs)
NS.NS_base.__init__ = init

_calc = NS.NS_base.calculateSolution
def calc(self, *a, **k):
    r = _calc(self, *a, **k)
    comm = Comm.get().comm.tompi4py()
    rows = []
    for mi, mod in enumerate(self.modelList):
        lm = mod.levelModelList[-1]
        for ci in range(lm.nc):
            dm = lm.u[ci].femSpace.dofMap
            offsets = dm.dof_offsets_subdomain_owned
            nOwned = int(offsets[comm.rank + 1] - offsets[comm.rank]) if comm.size > 1 else lm.u[ci].dof.shape[0]
            x = getattr(dm, "lagrangeNodesArray", lm.mesh.nodeArray)[:nOwned, :3]
            rows.append(np.hstack([np.full((nOwned, 1), 100 * mi + ci), x, lm.u[ci].dof[:nOwned, None]]))
    gathered = comm.gather(np.vstack(rows), root=0)
    if comm.rank == 0:
        a = np.vstack(gathered)
        np.savetxt("u.csv", a[np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))], delimiter=",", header="key,x,y,z,u")
        with open("models.csv", "w") as f:
            for mi, mod in enumerate(self.modelList):
                lm = mod.levelModelList[-1]
                for ci in range(lm.nc):
                    f.write("%d,%s\n" % (100 * mi + ci, lm.name))
    return r
NS.NS_base.calculateSolution = calc

FT = os.environ.get("FT", "0.1")
if SO:
    sys.argv = ['parun', SO, '-l' + os.environ.get('LOGL', '1'), '-C', os.environ.get("OPTS", "")]
else:
    sys.argv = ['parun', '--TwoPhaseFlow', DECK, '-l1', '-C',
                'final_time=%s dt_output=%s %s' % (FT, FT, os.environ.get("OPTS", "he=0.1"))]
runpy.run_path(shutil.which('parun'), run_name='__main__')
