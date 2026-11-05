"""vortex2D (level set + redistancing + VOF + mass correction) with a per-rank archive, stopped and hot
started; the driver for test_subdomain_view_restart.py (C7.1 Phase D).

    mpiexec -n R python restart_views_vortex.py OUTDIR MODE N MESH GRID DOCUMENT

MODE      stop  run to step N and stop, the archive left in OUTDIR
          hot   hot start from OUTDIR's archive at step N and run on to step N + 4
MESH      views   the hot start builds the mesh from the archive's subdomain views
          global  the hot start generates and partitions the mesh, as before Phase D
GRID      generated (each rank generates its piece of the box) or global (the whole mesh is generated
          and partitioned by partitionNodes)
DOCUMENT  global (layout B) or per-rank (layout C)

Rank 0 prints ``views=<0|1>``: whether the hot start's mesh came from the views. The PETSc solvers are
the hotstart_measure harness's parallel ones (GMRES + ASM/LU, tight tolerances).
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
vortex = os.path.join(here, "..", "levelset", "vortex2D")
outdir, mode, N, mesh_from, grid, document = sys.argv[1:7]
N = int(N)
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
sys.path.insert(0, vortex)

import vortex2D  # noqa: E402
vortex2D.parallel = True
vortex2D.domain.MeshOptions.regularGrid = grid
from petsc4py import PETSc  # noqa: E402
_opts = PETSc.Options()
for _pre in ("ncls_", "rdls_", "vof_", "mcorr_"):
    _opts[_pre + "ksp_type"] = "gmres"
    _opts[_pre + "ksp_rtol"] = "1e-12"
    _opts[_pre + "ksp_atol"] = "1e-14"
    _opts[_pre + "pc_type"] = "asm"
    _opts[_pre + "pc_asm_overlap"] = "1"
    _opts[_pre + "sub_pc_type"] = "lu"
vortex2D.soname = "views"

import ls_vortex_2d_p, redist_vortex_2d_p, vof_vortex_2d_p, ls_consrv_vortex_2d_p  # noqa: E402
import ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n  # noqa: E402
for _m in (ls_vortex_2d_n, redist_vortex_2d_n, vof_vortex_2d_n, ls_consrv_vortex_2d_n):
    _m.l_atol_res = 1e-14
    _m.linTolFac = 0.0
    _m.l_rtol_res = 0.0
import ls_vortex_2d_so as so  # noqa: E402
from proteus import NumericalSolution, default_s  # noqa: E402
from proteus.runtime import RunOptions  # noqa: E402

so.name = "views"
DT = so.DT
opts = RunOptions(gatherArchive=False, archiveDriver="per-rank", archiveDocument=document)
if mode == "stop":
    so.tnList = [i * DT for i in range(N + 1)]
else:
    so.tnList = [i * DT for i in range(N + 5)]
    opts.hotStart = True
    opts.hotStartTime = N * DT

if mode == "hot":
    opts.hotStartMesh = mesh_from          # NumericalSolution passes its open archive to generateMesh
sList = [default_s for _ in so.pnList]
pm = {"ls": (ls_vortex_2d_p, ls_vortex_2d_n),
      "redist": (redist_vortex_2d_p, redist_vortex_2d_n),
      "vof": (vof_vortex_2d_p, vof_vortex_2d_n),
      "ls_consrv": (ls_consrv_vortex_2d_p, ls_consrv_vortex_2d_n)}
order = [n.replace("_vortex_2d_p", "") for n, _ in so.pnList]
ns = NumericalSolution.NS_base(so, [pm[o][0] for o in order], [pm[o][1] for o in order], sList, opts)
mesh = ns.modelList[0].levelModelList[-1].mesh
if ns.comm.rank() == 0:
    print("views=%d" % int(bool(getattr(mesh.globalMesh, "fromSubdomainViews", False))))
    sys.stdout.flush()
ns.calculateSolution(so.name)
