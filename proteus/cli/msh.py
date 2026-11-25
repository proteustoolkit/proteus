"""``pro msh``: generate a model's mesh files, and nothing else.

Meshing with triangle, tetgen or gmsh is serial, and on a large mesh it
needs far more memory than one rank of the run. ``pro msh`` does it as a
job of its own -- one process, on a large-memory node if need be -- and
records what it made (see :mod:`proteus.cli.mesh`). ``pro run`` then finds
the mesh current and reads it instead of meshing on rank 0 of a large
allocation.

It is never required: ``pro run`` meshes first whenever the mesh is
missing or out of date.

Given a YMF spec, ``pro msh SPEC.ymf`` makes the meshes ``pro run
SPEC.ymf`` would solve on -- those of a region, which triangle or tetgen
generate; a box is meshed in memory -- and takes the YMF runner's options
(``--discretization``, ``--h``, ``--outdir``, ``--force``). See
:mod:`proteus.cli.ymfmesh`.
"""
import sys

from . import add_command

__all__ = ["add_parser", "run"]


def add_parser(subparsers):
    p = add_command(
        subparsers, "msh",
        help="generate the mesh files, without running the model",
        description="Generate a model's mesh files (serial) unless they are "
                    "already up to date. pro run would do the same before "
                    "running; this does it as a job of its own.")
    p.add_argument("inputs", nargs="+", metavar="MODEL.py",
                   help="so module, TwoPhaseFlow case module, or p and n modules; "
                        "or a YMF spec (SPEC.ymf), which takes pro run SPEC.ymf's "
                        "options")
    p.add_argument("-C", "--context", metavar="'NAME=VALUE ...'",
                   help="context options, as for pro run")
    p.add_argument("-o", "--context-file", metavar="FILE",
                   help="file of context options; -C entries override it")
    p.add_argument("-f", "--force", action="store_true",
                   help="regenerate even if the mesh is up to date")
    p.add_argument("-l", "--log-level", type=int, default=1, metavar="N",
                   help="log level; 0 turns the log file off (default 1)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="echo the log to the terminal")
    p.set_defaults(handler=run, pumi=False, petsc=None, petsc_file=None)
    return p


def plan_for(pList, nList):
    """``(domain, meshOptions, plan)`` for the mesh the run would use;
    ``plan`` is None when no mesh files are generated."""
    from proteus import MeshTools
    p, n = pList[0], nList[0]
    MeshTools.setDomainFromPhysics(p, n)
    meshOptions = MeshTools.chooseMeshOptions(p, n)
    return p.domain, meshOptions, MeshTools.meshFiles(p.domain, meshOptions)


def run(args):
    from .run import _check_inputs, _context_string, _load, _split
    error = _check_inputs(args)
    if error:
        print("pro msh: " + error, file=sys.stderr)
        return 2

    from proteus import runtime
    comm = runtime.init()
    if comm.size() > 1:
        if comm.isMaster():
            print("pro msh: meshing is serial; run it on one process, "
                  "not under mpiexec", file=sys.stderr)
        return 2

    from proteus import Context, Profiling
    # Comm.init() starts PETSc without setting it, and openLog asserts on it (pro run sets it the same way)
    Profiling.procID = comm.rank()
    if args.verbose:
        Profiling.verbose = True
    if args.log_level > 0:
        Profiling.openLog(_split(args.inputs[0])[1] + "_msh.log",
                          args.log_level)
    Context.contextOptionsString = _context_string(args)
    so, pList, nList, sList, two_phase = _load(args, Profiling.logEvent)

    domain, meshOptions, plan = plan_for(pList, nList)
    if plan is None:
        print("pro msh: nothing to generate: a {0} builds its mesh in memory "
              "or reads mesh files it is given".format(type(domain).__name__))
        return 0
    from . import mesh
    state, reason = mesh.state(plan)
    if state == "current" and not args.force:
        print("pro msh: up to date: " + mesh.manifest_path(plan))
        return 0
    print("pro msh: {0}: running {1}".format(reason or "forced", plan["command"]))
    mesh.make(plan, domain, meshOptions)
    print("pro msh: wrote " + mesh.manifest_path(plan))
    return 0
