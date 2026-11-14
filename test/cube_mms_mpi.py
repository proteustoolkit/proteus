#!/usr/bin/env python
"""Poisson on the unit square or cube with a C0 space Q_p on quadrilaterals or hexahedra, through NS_base
and OneLevelTransport; run by test_cube_mms.py, serially and under mpiexec. The counterpart of
simplex_mms_mpi.py (same problem, same checks).

    -div(grad u) = f,  u = g on the whole boundary (strong Dirichlet),

with the manufactured solution SOLUTION=smooth (a product of sines, shifted so that the boundary data is not
zero and has curvature along every face) or SOLUTION=poly (a polynomial of degree p in each coordinate: the
discrete solution must be exact). The errors are integrated here, on the elements each rank owns, with p+3
Gauss points per direction, from the model's dofs and the space's own tabulation, and summed over the ranks.

Also reported: the Jacobian's nonzero count against a brute-force count of the element couplings of the dof
map (on every rank), and, read back from the archive this run wrote, the dof vector, the dof points and the
sub-cell view of the space (cells, total volume).

Environment: ND (2|3), P, NODES (gll|equispaced), NN (nodes per direction), SOLUTION (smooth|poly),
PART_TYPE (node|element), NLAYERS. Rank 0 prints one line ``RESULT {json}``.
"""
import json
import os

from proteus import runtime
comm = runtime.start()

import numpy as np
from petsc4py import PETSc
from proteus import (CubeSpaces, Domain, FemTools, MeshTools, NumericalFlux, NumericalSolution, Quadrature,
                     TransportCoefficients, default_s, defaults)
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.TimeIntegration import NoIntegration
from proteus.Transport import OneLevelTransport

nd = int(os.environ.get("ND", "2"))
P = int(os.environ.get("P", "3"))
nodes = os.environ.get("NODES", "gll")
nn = int(os.environ.get("NN", "5"))
solution = os.environ.get("SOLUTION", "smooth")
part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "node")]
layers = int(os.environ.get("NLAYERS", "1"))

K = 1.5 * np.pi   # sin(K (x + 0.2)) is neither zero nor linear on the boundary
S = 0.2
C = np.array([1.0, -0.7, 0.4])[:nd]


def u_exact(x):
    """x (..., >= nd)"""
    if solution == "poly":
        return (0.3 + x[..., :nd] @ C) ** P + 0.5 * np.prod(x[..., :nd], axis=-1) ** P
    return np.prod(np.sin(K * (x[..., :nd] + S)), axis=-1)


def grad_u_exact(x):
    g = np.zeros(x.shape[:-1] + (nd,))
    if solution == "poly":
        g += (P * (0.3 + x[..., :nd] @ C) ** (P - 1))[..., None] * C
        for i in range(nd):   # d/dx_i of 0.5 (x_1 .. x_nd)^P, a member of Q_P that is not in P_P
            g[..., i] += 0.5 * P * x[..., i] ** (P - 1) * np.prod(np.delete(x[..., :nd], i, axis=-1), axis=-1) ** P
        return g
    s, c = np.sin(K * (x[..., :nd] + S)), np.cos(K * (x[..., :nd] + S))
    for i in range(nd):
        g[..., i] = K * c[..., i] * np.prod(np.delete(s, i, axis=-1), axis=-1)
    return g


def source(x):
    """f = -laplacian(u), evaluated at one point (the coefficient class calls it point by point)."""
    x = np.asarray(x)
    if solution == "poly":
        lap = P * (P - 1) * (0.3 + x[:nd] @ C) ** (P - 2) * (C @ C) if P >= 2 else 0.0
        if P >= 2:
            for i in range(nd):
                lap += 0.5 * P * (P - 1) * x[i] ** (P - 2) * np.prod(np.delete(x[:nd], i)) ** P
        return -lap
    return nd * K * K * np.prod(np.sin(K * (x[:nd] + S)))


def identity(x):
    return np.eye(nd)


def dirichlet(x, flag):
    if flag != 0:  # every exterior face of the generated box mesh has a positive flag
        return lambda x, t: float(u_exact(np.asarray(x)))
    return None


class Exact(object):
    def uOfXT(self, x, t):
        return float(u_exact(np.asarray(x)))


L = (1.0,) * nd
domain = Domain.RectangularDomain(L=L, x=(0.0,) * nd, name="box")
o = domain.MeshOptions
o.nnx = o.nny = o.nnz = o.nn = nn
o.triangleFlag = 0
o.quad = nd == 2
o.hex = nd == 3
o.parallelPartitioningType = part
o.nLayersOfOverlapForParallel = layers

p = defaults.Physics_base(nd=nd, name="c8_cube_mms")
p.domain = domain
p.LevelModelType = OneLevelTransport
p.coefficients = TransportCoefficients.PoissonEquationCoefficients({0: identity}, {0: source}, 1, nd, l2proj=[False])
p.initialConditions = {0: type("Zero", (), {"uOfXT": lambda self, x, t: 0.0})()}
p.dirichletConditions = {0: dirichlet}
p.advectiveFluxBoundaryConditions = {}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.analyticalSolution = {0: Exact()}

n = defaults.Numerics_base()
n.timeIntegration = NoIntegration
n.femSpaces = {0: lambda mesh, nd_: FemTools.C0_AffinePolynomialOnCube(mesh, nd_, P, nodes)}
# CubeGaussQuadrature takes the number of points per direction: P + 2 is exact to degree 2P + 3 in each
n.elementQuadrature = Quadrature.CubeGaussQuadrature(nd, P + 2)
n.elementBoundaryQuadrature = Quadrature.CubeGaussQuadrature(nd - 1, P + 2)
n.quad = nd == 2
n.hex = nd == 3
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.maxNonlinearIts = 3
n.fullNewtonFlag = True
n.tolFac = 0.0
n.nl_atol_res = 1.0e-10
n.matrix = SparseMatrix
# the same path serially and in parallel: strong Dirichlet dofs kept in the system, a direct solve
n.numericalFluxType = NumericalFlux.StrongDirichletFactory({}, keepDOFs=True)
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linearSmoother = None
n.linearSolverConvergenceTest = "r-true"
n.linTolFac = 0.0
n.l_atol_res = 1.0e-13
n.subgridError = None
n.shockCapturing = None
options = PETSc.Options()
options.setValue("ksp_type", "gmres")
options.setValue("ksp_rtol", 1.0e-14)
options.setValue("ksp_atol", 1.0e-15)
options.setValue("pc_type", "lu")
options.setValue("pc_factor_mat_solver_type", os.environ.get("LU", "superlu_dist" if comm.size() > 1 else "superlu"))
n.parallelPartitioningType = part
n.nLayersOfOverlapForParallel = layers

so = defaults.System_base(name="c8_cube_mms", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
ns.calculateSolution("c8_cube_mms")

model = ns.modelList[0].levelModelList[-1]
space = model.u[0].femSpace
mesh = model.mesh
dof = np.asarray(model.u[0].dof)
l2g = np.asarray(space.dofMap.l2g)
en = np.asarray(mesh.elementNodesArray)
X = np.asarray(mesh.nodeArray)
nOwned = mesh.nElements_owned
g1, w1 = np.polynomial.legendre.leggauss(P + 3)
xi = np.array(np.meshgrid(*([g1] * nd), indexing="ij")).reshape(nd, -1).T
w = np.prod(np.array(np.meshgrid(*([w1] * nd), indexing="ij")).reshape(nd, -1), axis=0)
tab = space.referenceFiniteElement.localFunctionSpace.tabulate(xi, 1)
corners = 2.0 * np.array(CubeSpaces.CORNERS[nd]) - 1.0
e = np.arange(nOwned)
Xe = X[en[e]]
xq = np.einsum("qv,evc->eqc", CubeSpaces.vertex_weights(nd, xi), Xe)
# parallelogram / parallelepiped elements: x = a + J xi, J = sum_v (X_v - a) c_v^T / 2^nd
centre = Xe[:, :, :nd].mean(axis=1)
J = np.einsum("evi,vj->eij", Xe[:, :, :nd] - centre[:, None, :], corners) / 2.0 ** nd
affine_defect = float(np.abs(Xe[:, :, :nd] - centre[:, None, :] - np.einsum("eij,vj->evi", J, corners)).max()) if nOwned > 0 else 0.0
detJ = np.abs(np.linalg.det(J))
Jinv = np.linalg.inv(J)
uh = np.einsum("qi,ei->eq", tab.values, dof[l2g[e]])
grad_ref = np.einsum("qid,ei->eqd", tab.grads, dof[l2g[e]])
grad = np.einsum("eqd,edk->eqk", grad_ref, Jinv)                         # grad_x = J^-T grad_xi
err2 = float(np.einsum("q,e,eq->", w, detJ, (uh - u_exact(xq)) ** 2))
gerr2 = float(np.einsum("q,e,eqk->", w, detJ, (grad - grad_u_exact(xq)) ** 2))
linf = float(np.abs(uh - u_exact(xq)).max()) if nOwned > 0 else 0.0
from mpi4py import MPI
mc = comm.comm.tompi4py()
err2, gerr2 = mc.allreduce(err2, MPI.SUM), mc.allreduce(gerr2, MPI.SUM)
linf = mc.allreduce(linf, MPI.MAX)
nElements = mc.allreduce(int(nOwned), MPI.SUM)
# sparsity: the Jacobian's pattern is exactly the element couplings of the dof map (no flux stencil here)
rows = np.repeat(l2g, l2g.shape[1], axis=1).ravel().astype(np.int64)
cols = np.tile(l2g, (1, l2g.shape[1])).ravel().astype(np.int64)
nnz_brute = int(len(np.unique(rows * space.dofMap.nDOF + cols)))
nnz_model = int(model.nnz)
nnz_equal_on_every_rank = bool(comm.comm.tompi4py().allreduce(int(nnz_model == nnz_brute), MPI.MIN))
csr_dtype = str(model.rowptr.dtype)
# the archive: the complete dof vector on the sub-cell view of the space (written by the space class)
archive = {}
if comm.rank() == 0:
    import h5py
    suffix = "_c0q%d_%s" % (P, nodes)
    with h5py.File("c8_cube_mms.h5", "r") as h5:
        s2g = np.asarray(space.dofMap.subdomain2global)
        ua = h5["u0_t1"][:]
        xa = h5["nodes" + suffix + "0"][:]
        ea = h5["elements" + suffix + "0"][:]
    nSub = CubeSpaces.subcells(nd, P).shape[0]
    # the sub-cells are parallelograms / parallelepipeds: the edges from vertex 0 to its neighbours 1, 3 (, 4)
    edges = (xa[ea[:, [1, 3, 4][:nd]]] - xa[ea[:, :1]])[:, :, :nd]
    archive = dict(archive_u=float(np.abs(ua[s2g] - dof).max()),
                   archive_x=float(np.abs(xa[s2g] - space.dofMap.lagrangeNodesArray).max()),
                   archive_nDOF=int(len(ua)), archive_cells=int(ea.shape[0]), archive_nSub=int(nSub),
                   archive_volume=float(np.abs(np.linalg.det(edges)).sum()), archive_vertices=int(ea.shape[1]),
                   archive_min_cell=float(np.abs(np.linalg.det(edges)).min()))
if comm.rank() == 0:
    print("RESULT " + json.dumps(dict(nd=nd, p=P, nodes=nodes, nn=nn, solution=solution, affine_defect=affine_defect,
                                      ranks=comm.size(), L2=err2 ** 0.5, H1=gerr2 ** 0.5, Linf=linf,
                                      nDOF=int(space.dofMap.nDOF_all_processes), nElements=nElements,
                                      h=1.0 / (nn - 1), nnz_model=nnz_model, nnz_brute=nnz_brute,
                                      nnz_equal_on_every_rank=nnz_equal_on_every_rank,
                                      csr_dtype=csr_dtype, space_class=type(space).__name__, **archive)), flush=True)
