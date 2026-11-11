"""
Module for controlling MPI

.. inheritance-diagram:: proteus.Comm
   :parts: 1
"""
import ctypes
import sys

# scotch has been compiled with SCOTCH_PTHREAD
# so MPI_Init() must be called with MPI_THREAD_MULTIPLE, or MPI_Init_thread() must be called.
# at import time, mpi4py initializes the MPI execution environment calling MPI_Init_thread().
# mpi4py must be imported before petsc4py otherwise petsc4py initializes MPI with MPI_Init().
from mpi4py import MPI
try:
    import petsc4py
except ImportError:
    #proteus's own solvers need only MPI; see petsc_optional
    petsc4py = None
from .Profiling import logEvent


# Special workaround for broken MPI on certain Cray systems
from . import config
from . import _threads
mpi_preload_libs=[]
for lib in config.PROTEUS_PRELOAD_LIBS:
    mpi_preload_libs.append(ctypes.CDLL(lib,mode=ctypes.RTLD_GLOBAL))

# One thread per rank under MPI.  import proteus already did this when the launcher announces
# its size; this covers a launcher that does not, and numpy imported before proteus.
_thread_warning = _threads.after_mpi_init(MPI.COMM_WORLD.Get_size())
if _thread_warning:
    logEvent(_thread_warning, 0)

comm = None
argv = sys.argv
petscInitialized = False

def _set_reproducible_vector_defaults():
    """Make PETSc's multi-vector dot product bitwise reproducible from run to run.

    PETSc >= 3.21 computes VecMDot, which GMRES with classical Gram-Schmidt calls once per
    iteration, with one BLAS gemv over its Krylov vectors (-vec_mdot_use_gemv, default on).
    The sums that OpenBLAS's gemv forms depend on the alignment of those vectors, which
    malloc settles from the heap's history, so the same linear solve can differ in the last
    bit from one run to the next (from the tenth Krylov vector on; CLSVOF's GMRES showed it as
    a damBreak that differed in 1 run of 20 on two ranks). The plain C kernel does not have
    the dependence. A value given on the command line or in PETSC_OPTIONS is kept.
    """
    from petsc4py import PETSc
    OptDB = PETSc.Options()
    if not OptDB.hasName("vec_mdot_use_gemv"):
        OptDB.setValue("vec_mdot_use_gemv", False)
    # PETSc reads it when the first vector is made; a run that makes none (a shim, a spec check) would end with
    # "WARNING! There are options you set that were not used!", so read it here to mark it used
    OptDB.getBool("vec_mdot_use_gemv", False)

def init():
    global comm, petscInitialized
    if petsc4py is None:
        new_comm = Comm()
        if not comm:
            comm = new_comm
        return new_comm
    if not petscInitialized:
        petsc4py.init(argv)
        petscInitialized = True
    else:
        from petsc4py import PETSc
        OptDB=PETSc.Options()
        narg = len(argv)
        for i,s in enumerate(argv):
            if len(s) > 0 and s[0] == '-':
                name = s
                if i+1 < narg and  argv[i+1][0] != '-':
                    value = argv[i+1]
                    OptDB.setValue(name,value)
        if not OptDB.hasName("options_left"):
            OptDB.setValue("options_left",False)
    _set_reproducible_vector_defaults()
    petsc4py.PETSc.Log.Stage('proteus').push()
    new_comm = Comm()
    if not comm:
        comm = new_comm
    return new_comm

def get():
    if comm == None:
        logEvent("Comm.get called before init, init is being called for you.", 3)
        return init()
    return comm

class _MPIWorld(object):
    """The part of a PETSc communicator's interface proteus uses, over mpi4py.

    Callers write comm.comm.rank, comm.comm.size, comm.comm.Barrier() and
    comm.comm.tompi4py(); with petsc4py comm.comm is PETSc.COMM_WORLD, which
    provides those, and without it this does.
    """
    def __init__(self, mpi_comm):
        self._mpi = mpi_comm
        self.rank = mpi_comm.Get_rank()
        self.size = mpi_comm.Get_size()

    def Barrier(self):
        return self._mpi.Barrier()

    def tompi4py(self):
        return self._mpi

class Comm(object):
    """Proteus wrapper around PETSc/MPI communicators

    This is a very simple class that provides compatibility with older Proteus comm objects.
    """

    def __init__(self):
        if petsc4py is None:
            self.comm = _MPIWorld(MPI.COMM_WORLD)
        else:
            from petsc4py import PETSc
            self.comm = PETSc.COMM_WORLD
        self.mpi4py_comm = self.comm.tompi4py()
    def isInitialized(self):
        return True

    def rank(self):
        """Return position in this communicator (0-indexed)"""
        return self.comm.rank

    def size(self):
        """Return number of processes on this communicator"""
        return self.comm.size

    def barrier(self):
        """Block all processes on this communicator until all have called this method"""
        return self.comm.Barrier()

    def isMaster(self):
        """Return true if this process is the 'master' for this communicator"""
        return self.comm.rank == 0

    def beginSequential(self, ng=1):
        """Begin a block of code to be completed sequentially by each process
        """

        if ng != 1:
            raise NotImplementedError("Only ng = 1 allowed")

        for i in range(self.comm.rank):
            self.comm.Barrier()
        return

    def endSequential(self, ng=1):
        """End a block of code to be completed sequentially by each process
        """

        if ng != 1:
            raise NotImplementedError("Only ng = 1 allowed")

        for i in range(self.comm.size - self.comm.rank - 1):
            self.comm.Barrier()
        return

    def globalSum(self, value):
        from mpi4py import MPI
        rvalue = self.mpi4py_comm.allreduce(sendobj = value,
                                            op = MPI.SUM)
        return rvalue
    
    def globalMax(self, value):
        from mpi4py import MPI
        rvalue = self.mpi4py_comm.allreduce(sendobj = value,
                                            op = MPI.MAX)
        return rvalue

    def globalMin(self, value):
        from mpi4py import MPI
        rvalue = self.mpi4py_comm.allreduce(sendobj = value,
                                            op = MPI.MIN)
        return rvalue
    
def globalSum(value):
    return comm.globalSum(value)

def globalMax(value):
    return comm.globalMax(value)

def globalMin(value):
    return comm.globalMin(value)
