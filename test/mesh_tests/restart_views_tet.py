"""A VOF blob on tetrahedra with a per-rank archive, stopped and hot started; the driver for
test_subdomain_view_restart.py's tetrahedra case (C7.1 round 3; the Phase D test is vortex2D, 2D).

    mpiexec -n R python restart_views_tet.py OUTDIR MODE N MESH DOCUMENT

MODE      stop  run to step N and stop, the archive left in OUTDIR
          hot   hot start from OUTDIR's archive at step N and run on to step N + 4
MESH      views   the hot start builds the mesh from the archive's subdomain views
          global  the hot start reads and partitions the mesh, as before Phase D
DOCUMENT  global (layout B) or per-rank (layout C)

The mesh is importer_meshes/gmsh_cube.msh (341 nodes, 1125 tetrahedra) through a use_gmsh PLC domain,
partitioned by the graph partitioner (pin it with -mat_partitioning_type current). mprans VOF, a
Gaussian blob carried by the constant velocity (1, 1, 1) / 4, backward Euler with a fixed step, P1,
SUPG, a direct solve. Rank 0 prints ``views=<0|1>``: whether the hot start's mesh came from the views.
"""
import os
import shutil
import sys

here = os.path.dirname(os.path.abspath(__file__))
outdir, mode, N, mesh_from, document = sys.argv[1:6]
N = int(N)
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)

from proteus import runtime  # noqa: E402

comm = runtime.init("-vof_ksp_type preonly -vof_pc_type redundant -vof_redundant_pc_type lu")
if comm.isMaster() and not os.path.exists("gmsh_cube.msh"):
    shutil.copy(os.path.join(here, "importer_meshes", "gmsh_cube.msh"), "gmsh_cube.msh")
comm.barrier()

import numpy as np  # noqa: E402

from proteus import (Domain, MeshTools, NumericalSolution, SpatialTools, StepControl, TimeIntegration,  # noqa: E402
                     default_s, defaults)
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis  # noqa: E402
from proteus.LinearAlgebraTools import SparseMatrix  # noqa: E402
from proteus.LinearSolvers import KSP_petsc4py  # noqa: E402
from proteus.mprans import VOF  # noqa: E402
from proteus.NonlinearSolvers import Newton  # noqa: E402
from proteus.Quadrature import SimplexGaussQuadrature  # noqa: E402

domain = Domain.PiecewiseLinearComplexDomain(name="gmsh_cube")
SpatialTools.Cuboid(domain, dim=[1.0, 1.0, 1.0], coords=[0.5, 0.5, 0.5])
SpatialTools.assembleDomain(domain)
domain.geofile = "gmsh_cube"           # the stored .msh beside it is read; gmsh does not run
options = MeshTools.getMeshOptions(domain)
options.use_gmsh = True
options.genMesh = False
options.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
options.nLayersOfOverlapForParallel = 1

V = np.array([0.25, 0.25, 0.25])


class _Flow(object):
    """What VOF reads from a flow model; there is none here and no solid."""


class Coefficients(VOF.Coefficients):
    def attachModels(self, modelList):
        VOF.Coefficients.attachModels(self, modelList)
        m = modelList[self.modelIndex]
        flow = _Flow()
        flow.ghost_penalty_constant = 0.0
        flow.phi_s = np.full(m.mesh.nodeArray.shape[0], 1.0e10)
        flow.ebqe_phi_s = np.full(m.ebqe[('u', 0)].shape, 1.0e10)
        flow.q_phi_solid = np.full(m.q[('u', 0)].shape, 1.0e10)
        self.flowCoefficients = flow
        for key, q in (("q_v", m.q), ("ebqe_v", m.ebqe)):
            v = np.zeros(q[('u', 0)].shape + (3,))
            v[...] = V
            setattr(self, key, v)


class Blob(object):
    def uOfXT(self, x, t):
        return np.exp(-(((x[0] - 0.35) ** 2 + (x[1] - 0.35) ** 2 + (x[2] - 0.35) ** 2) / 0.15 ** 2))


def inflow(x, flag):
    if min(x[0], x[1], x[2]) <= 1.0e-8:
        return lambda x, t: 0.0


p = defaults.Physics_base(nd=3, name="vof")
p.domain = domain
p.LevelModelType = VOF.LevelModel
p.coefficients = Coefficients(LS_model=None, V_model=None, RD_model=None, ME_model=0,
                              checkMass=False, epsFact=0.0, STABILIZATION_TYPE=0)
p.initialConditions = {0: Blob()}
p.dirichletConditions = {0: inflow}
p.advectiveFluxBoundaryConditions = {0: lambda x, flag: None}
p.diffusiveFluxBoundaryConditions = {0: {}}
p.fluxBoundaryConditions = {0: 'outFlow'}

n = defaults.Numerics_base()
n.timeIntegration = TimeIntegration.BackwardEuler
n.stepController = StepControl.FixedStep
n.DT = 0.05
n.femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
n.elementQuadrature = SimplexGaussQuadrature(3, 3)
n.elementBoundaryQuadrature = SimplexGaussQuadrature(2, 3)
n.multilevelNonlinearSolver = n.levelNonlinearSolver = Newton
n.fullNewtonFlag = True
n.numericalFluxType = VOF.NumericalFlux
n.shockCapturing = VOF.ShockCapturing(p.coefficients, 3, shockCapturingFactor=0.0, lag=False)
n.subgridError = VOF.SubgridError(p.coefficients, 3)
n.matrix = SparseMatrix
n.multilevelLinearSolver = n.levelLinearSolver = KSP_petsc4py
n.linear_solver_options_prefix = 'vof_'
n.tolFac = 0.0
n.nl_atol_res = 1.0e-12
n.l_atol_res = 1.0e-13
n.linTolFac = 0.0
n.maxNonlinearIts = 10
n.parallelPartitioningType = options.parallelPartitioningType
n.nLayersOfOverlapForParallel = options.nLayersOfOverlapForParallel

so = defaults.System_base(name="tetviews", pnList=[(p, n)], sList=[default_s])
opts = runtime.RunOptions(gatherArchive=False, archiveDriver="per-rank", archiveDocument=document)
if mode == "stop":
    so.tnList = [i * n.DT for i in range(N + 1)]
else:
    so.tnList = [i * n.DT for i in range(N + 5)]
    opts.hotStart = True
    opts.hotStartTime = N * n.DT
    opts.hotStartMesh = mesh_from          # NumericalSolution passes its open archive to generateMesh
runtime.start(log="proteus_default.log", log_level=3)
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
mesh = ns.modelList[0].levelModelList[-1].mesh
if ns.comm.rank() == 0:
    print("views=%d" % int(bool(getattr(mesh.globalMesh, "fromSubdomainViews", False))), flush=True)
print("rank=%d elements=%d nodes=%d" % (ns.comm.rank(), mesh.nElements_global, mesh.nNodes_global), flush=True)
ns.calculateSolution(so.name)
