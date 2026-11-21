"""What a Proteus run needs before it starts, as plain functions and data.

``parun`` and ``iproteus`` each build these from an optparse parser at
import time. This module holds the same things without a parser, so that
``pro run``, a notebook, ``ADRProblem`` and the tests can share them:

* :class:`RunOptions`, the options :class:`~proteus.NumericalSolution.NS_base`
  reads from its ``opts`` argument;
* :func:`petsc_argv`, PETSc's command line from options and options files;
* :func:`init`, which starts MPI and PETSc with that command line;
* :func:`start`, :func:`init` once, with the log open, for a library caller.
* :func:`abort_on_uncaught_exception`, which makes an exception that one
  rank raises and nothing catches end the whole MPI job.

Nothing here imports PETSc at module level, so importing it never starts
PETSc: options that PETSc reads only at start-up (``-log_view``,
``-malloc_debug``) take effect only if :func:`init` runs before anything
else imports ``petsc4py.PETSc``.
"""
import math
import sys
from dataclasses import dataclass

__all__ = ["RunOptions", "petsc_argv", "init", "start", "abort_on_uncaught_exception", "drain_before_abort"]


@dataclass
class RunOptions:
    """The options :class:`~proteus.NumericalSolution.NS_base` reads.

    The field names are the attribute names ``NS_base`` reads, which are
    parun's optparse ``dest`` names, and the defaults are parun's. Every
    instance is independent, unlike ``iproteus.opts``, which is one object
    shared by everything that imports it.
    """
    #: directory for archives and logs; '' is the working directory (-D)
    dataDir: str = ''
    #: archive in ASCII text files instead of HDF5 (-T)
    useTextArchive: bool = False
    #: view the solution on every mesh level (-L)
    viewLevels: bool = False
    #: view the mesh (--viewMesh)
    viewMesh: bool = False
    #: stop after each nonlinear solve (-w)
    wait: bool = False
    #: cache archive writes (-c)
    cacheArchive: bool = False
    #: gather the per-rank archives at the end (-G)
    gatherArchive: bool = False
    #: one archive per subdomain instead of a global one (-s)
    subdomainArchives: bool = False
    #: synchronize archive metadata globally (parun -n clears it)
    global_sync: bool = True
    #: the archive's mode, driver and document layout (--archive-mode,
    #: --archive-driver, --archive-document); None is Archiver's default
    archiveMode: str = None
    archiveDriver: str = None
    archiveDocument: str = None
    #: restart from the archive (-H)
    hotStart: bool = False
    #: what a hot start continues (--hot-start-mode): "continue" resumes the
    #: numerical state (lagged shock capturing, BDF history) and needs the step
    #: policy of the archived run; "restart" continues the solution only and
    #: rebuilds that state from it, so the policy may change
    hotStartMode: str = "continue"
    #: write the solver state with every N-th archived frame (--state-every); 0
    #: writes it at the final step and at a checkpoint only
    stateEvery: int = 0
    #: stop at a step boundary with a checkpoint before this much wall time has
    #: passed ("HH:MM:SS", or "scheduler"); None: no limit (--wall-limit)
    wallLimit: str = None
    #: time to restart from (-t)
    hotStartTime: float = math.inf
    #: write the velocity post-processor (-B)
    writeVPP: bool = False
    #: read a mesh partitioned ahead of time (-F)
    generatePartitionedMeshFromFiles: bool = False
    #: save the degrees of freedom (-S)
    save_dof: bool = False


def _read_options_file(path):
    """The whitespace-separated words of a PETSc options file, without
    ``#`` comments."""
    words = []
    with open(path) as f:
        for line in f:
            words += line.split('#', 1)[0].split()
    return words


def petsc_argv(options=None, options_file=None, program=None):
    """PETSc's command line: the options file's words, then ``options``.

    ``options`` is a string (``"-ksp_type gmres -pc_type lu"``) or a list
    of words. Later options win in PETSc, so the command line overrides the
    file. (parun, given both, silently drops ``-P`` and keeps only ``-O``.)
    """
    if program is None:
        # PETSc skips argv[0] as the program name. Under `python -c` it is
        # "-c", and an empty sys.argv has none; either would otherwise
        # reach PETSc as (or in place of) an option.
        program = sys.argv[0] if sys.argv and sys.argv[0] and not sys.argv[0].startswith('-') else 'proteus'
    argv = [program]
    if options_file is not None:
        argv += _read_options_file(options_file)
    if options:
        argv += options.split() if isinstance(options, str) else list(options)
    return argv


def init(options=None, options_file=None):
    """Start MPI and PETSc with the given PETSc options; return the Comm.

    Safe to call more than once. The first call that reaches PETSc starts
    it with these options; a later call adds ``-name value`` pairs to
    PETSc's options database (see :func:`proteus.Comm.init`).
    """
    from proteus import Comm
    Comm.argv = petsc_argv(options, options_file)
    return Comm.init()


def start(log="proteus.log", log_level=1, abort_program=None):
    """Start Proteus once for a caller that has no command line; return the Comm.

    If nothing has started Proteus, :func:`init` it with no PETSc options
    (``Comm.get()`` would hand PETSc all of ``sys.argv``); set
    ``Profiling.procID`` if it is unset. Then open ``log`` in the working directory at
    ``log_level`` unless a log is already open, or ``log`` is None. This is
    what importing :mod:`proteus.iproteus` did, without its shared ``opts``.

    A script that runs under ``mpiexec`` directly, not through ``parun`` or
    ``pro``, has no :func:`abort_on_uncaught_exception` unless it asks:
    ``abort_program="name"`` installs it (see there), so an exception on one
    rank ends the job instead of leaving the other ranks in a collective.
    """
    from proteus import Comm, Profiling
    comm = Comm.comm
    if comm is None:
        comm = init()
    if Profiling.procID is None:
        # Comm.get() starts PETSc without setting it
        Profiling.procID = comm.rank()
    if log is not None and Profiling.logFile is None:
        Profiling.openLog(log, log_level)
    if abort_program is not None:
        abort_on_uncaught_exception(abort_program)
    return comm


#: Seconds to let a launcher forward what the ranks wrote before ``MPI_Abort``.
#: MPICH's hydra kills its proxies on abort and can exit before it has forwarded
#: the ranks' output (empty stderr, exit 1: ``abortWith`` in partitioning.cpp,
#: macOS 2026-11-15; this hook, ceds-model nightly 2026-11-16 at load 31).
#: partitioning.cpp's ``abortWith`` waits the same 0.2 s.
ABORT_DRAIN_SECONDS = 0.2


def drain_before_abort(seconds=None):
    """Flush every output buffer, then wait for the launcher to forward it.

    Flushes Python's stdout and stderr and, through libc, every C stdio
    stream (``fflush(NULL)``: PETSc and the extensions print through it),
    then sleeps :data:`ABORT_DRAIN_SECONDS`. Call it right before
    ``MPI_Abort``, when the run is ending anyway. Never raises.
    """
    import time
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass
    time.sleep(ABORT_DRAIN_SECONDS if seconds is None else seconds)


def abort_on_uncaught_exception(program):
    """Make an exception that no code catches end the whole MPI job, not one rank.

    Installs a :data:`sys.excepthook`. The ranks that did not raise are
    waiting, or soon will be, in a collective for the one that did, and they
    would wait until the job's wall time runs out: on LONI QB (job 1173802,
    4096 ranks) one rank raised a ValueError in its first archive write and
    the other 4095 hung for 2.7 h. The hook writes ``<program>: uncaught
    exception on rank r of n`` and the traceback to stderr and to the log,
    flushes both and, with more than one rank, calls MPI ``Abort(1)``. With
    one rank, or before MPI starts, the process exits 1 as Python's own hook
    would have it. A failure that every rank reports together (a
    StepTimeError, a failed system step) is for the driver to catch and exit
    on, without an abort.

    The communicator is looked up when the hook runs, so it can be installed
    before MPI starts. Used by ``parun`` and ``pro``.
    """
    def hook(excType, excValue, excTraceback):
        import traceback
        from proteus import Comm, Profiling
        comm = Comm.comm
        rank, size = (comm.rank(), comm.size()) if comm is not None else (0, 1)
        text = "{0}: uncaught exception on rank {1} of {2}{3}\n{4}".format(
            program, rank, size, "; aborting the MPI job" if size > 1 else "",
            "".join(traceback.format_exception(excType, excValue, excTraceback)))
        for stream in (sys.stdout, sys.stderr):
            try:
                if stream is sys.stderr:
                    stream.write(text)
                stream.flush()
            except Exception:
                pass
        try:
            if Profiling.logFile is not None:
                Profiling.logEvent(text, level=0)
                Profiling.logFile.flush()
        except Exception:
            pass
        if size > 1:
            drain_before_abort()
            comm.comm.tompi4py().Abort(1)
    sys.excepthook = hook
    return hook
