#!/usr/bin/env python
"""A P1 Poisson solve on a mesh from one of the in-tree importers; run by test_importer_solves.py.

Usage: importer_solve_mpi.py KIND OUT

-Laplace(u) = -2 nd, u = |x|^2 on every boundary node whose tag is positive (so an
untagged exterior boundary shows up as a wrong solution), P1, one Newton step, a direct
solve (redundant LU). Rank 0 writes OUT.npz: the owned nodes' coordinates and values from
every rank, and the nodes per rank. KIND picks the importer:

- ``gmsh2d``, ``gmsh3d``: a PSLG/PLC domain with ``use_gmsh``, read from the stored
  ``importer_meshes/gmsh_square.msh`` / ``gmsh_cube.msh`` (the stored ymf mesh,
  ymf-import-gmsh's conversion of it). The stored .msh is the mesh; gmsh does not run.
- ``regular2d``, ``regular3d``: a box, ``generatePartitionedMeshFromFiles`` (parun -F)
  when the environment has F=1; X0 shifts the box from the origin.
- ``dmplex3d``: ``gmsh_cube.msh`` read by PETSc into a DMPlex, then the DMPlex importer.

The script copies what it reads into the working directory, which the caller makes.
"""
import os
import shutil
import sys

from proteus import runtime

# a direct solve, so the serial and partitioned answers differ by round-off only
comm = runtime.init("-imp_ksp_type preonly -imp_pc_type redundant -imp_redundant_pc_type lu")
runtime.start(log=None)

import numpy as np
from mpi4py import MPI

from proteus import Domain, MeshTools, NumericalSolution, default_s, defaults
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.LinearAlgebraTools import SparseMatrix
from proteus.LinearSolvers import KSP_petsc4py
from proteus.NonlinearSolvers import Newton
from proteus.NumericalFlux import Advection_DiagonalUpwind_Diffusion_SIPG_exterior
from proteus.Quadrature import SimplexGaussQuadrature
from proteus.TimeIntegration import NoIntegration
from proteus.TransportCoefficients import PoissonEquationCoefficients

HERE = os.path.dirname(os.path.abspath(__file__))
MESHES = os.path.join(HERE, "importer_meshes")
kind, out = sys.argv[1], sys.argv[2]
mpi = comm.comm.tompi4py()


def _copy(name):
    if comm.isMaster():
        shutil.copy(os.path.join(MESHES, name), name)
    mpi.barrier()


def square_pslg(name):
    d = Domain.PlanarStraightLineGraphDomain(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], vertexFlags=[1, 2, 3, 4],
        segments=[[0, 1], [1, 2], [2, 3], [3, 0]], segmentFlags=[1, 2, 3, 4], name=name)
    d.facets = [[[0, 1, 2, 3]]]
    d.facetFlags = [1]
    return d


def cube_plc(name):
    from proteus import SpatialTools as st
    d = Domain.PiecewiseLinearComplexDomain(name=name)
    st.Cuboid(d, dim=[1.0, 1.0, 1.0], coords=[0.5, 0.5, 0.5])
    st.assembleDomain(d)
    return d


generateFromFiles = False
if kind in ("gmsh2d", "gmsh3d"):
    nd = 2 if kind == "gmsh2d" else 3
    base = "gmsh_square" if nd == 2 else "gmsh_cube"
    _copy(base + ".msh")
    domain = square_pslg(base) if nd == 2 else cube_plc(base)
    domain.geofile = base          # what writeGeo sets; the .msh beside it is read, gmsh does not run
    options = MeshTools.getMeshOptions(domain)
    options.use_gmsh = True
    options.genMesh = False
elif kind in ("regular2d", "regular3d"):
    nd = 2 if kind == "regular2d" else 3
    x0 = [float(v) for v in os.environ.get("X0", "0,0,0").split(",")][:nd]
    domain = Domain.RectangularDomain(L=(1.0,) * nd, x=x0, name="box")
    options = domain.MeshOptions
    options.nnx = options.nny = options.nn = int(os.environ.get("NN", "9"))
    options.nnz = options.nn if nd == 3 else None
    options.triangleFlag = 1
    options.genMesh = True
    generateFromFiles = os.environ.get("F", "0") == "1"
elif kind == "dmplex3d":
    from petsc4py import PETSc
    nd = 3
    _copy("gmsh_cube.msh")
    plex = PETSc.DMPlex().createFromFile("gmsh_cube.msh", interpolate=True, comm=PETSc.COMM_SELF)
    domain = Domain.DMPlexDomain(plex)
    domain.name = "plex"
    options = domain.MeshOptions
else:
    raise SystemExit("unknown KIND %r" % (kind,))
options.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
options.nLayersOfOverlapForParallel = 0


def exact(x, t=0.0):
    return float(np.dot(x[:nd], x[:nd]))


def dirichlet(x, flag):
    if flag > 0:
        return exact


def a(x):
    return np.eye(nd).flat[:]


def f(x):
    return -2.0 * nd


p = defaults.Physics_base(nd=nd, name="importer_poisson")
p.domain = domain
p.coefficients = PoissonEquationCoefficients({0: a}, {0: f}, nc=1, nd=nd)
p.dirichletConditions = {0: dirichlet}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.diffusiveFluxBoundaryConditions = {0: {0: lambda x, flag: None}}
p.fluxBoundaryConditions = {0: 'setFlow'}

n = defaults.Numerics_base()
n.timeIntegration = NoIntegration
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(nd, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.maxNonlinearIts = 1
n.numericalFluxType = Advection_DiagonalUpwind_Diffusion_SIPG_exterior
n.matrix = SparseMatrix
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linear_solver_options_prefix = 'imp_'
n.tolFac = 0.0
n.nl_atol_res = 1.0e-10
n.linTolFac = 0.0
n.l_atol_res = 1.0e-12
n.parallelPartitioningType = options.parallelPartitioningType
n.nLayersOfOverlapForParallel = options.nLayersOfOverlapForParallel

opts = runtime.RunOptions(generatePartitionedMeshFromFiles=generateFromFiles)
so = defaults.System_base(name="importer_poisson", pnList=[(p, n)], sList=[default_s])
so.tnList = [0.0, 1.0]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution("importer_poisson")

model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
u = model.u[0]
owned = mesh.nNodes_owned
x = mpi.gather(np.array(mesh.nodeArray[:owned, :nd]), root=0)
v = mpi.gather(np.array(u.dof[:owned]), root=0)
nExterior = mpi.gather(int(((mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray]) > 0).sum()),
                       root=0)
nOwned = mpi.gather(int(owned), root=0)
if comm.isMaster():
    x = np.concatenate(x)
    v = np.concatenate(v)
    order = np.lexsort(x.T[::-1])
    err = np.abs(v - (x ** 2).sum(axis=1)).max()
    np.savez(out, x=x[order], u=v[order], owned=np.array(nOwned))
    print("RESULT ranks=%d nodes=%d owned=%s err=%.6e tagged_exterior=%s"
          % (comm.size(), len(v), nOwned, err, nExterior), flush=True)
