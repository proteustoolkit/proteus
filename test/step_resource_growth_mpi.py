#!/usr/bin/env python
"""test/TwoPhaseFlow/damBreakLevelSet.py (RANS2P, VOF, NCLS, RDLS, MCorr on the stored mesh) to t=FT, recording after
every time step, on each rank, what a leak per step would make grow:

  req    live MPI request handles (MPICH only, read from its request pool; -1 elsewhere)
  comm   live MPI communicator handles (MPICH only; -1 elsewhere)
  parvec live ParVec_petsc4py objects (ghosted vectors: each holds a PetscSF and its persistent MPI requests)

Run under mpiexec by test_step_resource_growth.py in an empty directory; leaves growth.<rank>.csv there (step, t, req,
comm, parvec). The archive is written every DT_OUTPUT so the HDF5 path is in every step range. LEAKY=1 is the control:
it puts back the scatterDOFArray of main f392a159, which cached a ghosted vector per id(array) and so kept one for every
temporary it was handed (VOF.FCTStep passes three per step).
"""
import ctypes, gc, glob, os, runpy, shutil, sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TwoPhaseFlow")
for f in [os.path.join(SRC, "damBreakLevelSet.py")] + glob.glob(os.path.join(SRC, "meshDambreak*")):
    shutil.copy(f, ".")

from mpi4py import MPI
from proteus import Comm
from proteus import NumericalSolution as NS
from proteus import Transport
from proteus.LinearAlgebraTools import ParVec_petsc4py


def _mpich_pools():
    """name -> function giving the live handles of that MPICH object pool, for the pools whose reading is validated
    here (allocating and freeing handles must move it by exactly that many); empty for any other MPI."""
    if "MPICH" not in MPI.Get_library_version():
        return {}
    lib = None
    for name in (None, "libpmpi.12.dylib", "libmpi.12.dylib", "libpmpi.so.12", "libmpi.so.12", "libmpi.so"):
        try:
            candidate = ctypes.CDLL(name) if name is None else ctypes.CDLL(
                name, mode=getattr(os, "RTLD_NOLOAD", 0) | ctypes.RTLD_GLOBAL)
            ctypes.c_int.in_dll(candidate, "MPIR_Request_mem")
            lib = candidate
            break
        except (OSError, ValueError):
            continue
    if lib is None:
        return {}

    def reader(symbol):
        # MPIR_Object_alloc_t: num_allocated and num_avail are its 9th and 10th ints (MPICH 4 and 5)
        v = (ctypes.c_int * 10).from_address(ctypes.addressof(ctypes.c_int.in_dll(lib, symbol)))
        return lambda: v[8] - v[9]

    def make_requests(n):
        buf = bytearray(8)
        return [MPI.COMM_SELF.Recv_init([buf, MPI.BYTE], source=0, tag=1) for _ in range(n)]

    def make_comms(n):
        return [MPI.COMM_SELF.Dup() for _ in range(n)]

    pools = {}
    for key, symbol, make in (("req", "MPIR_Request_mem", make_requests), ("comm", "MPIR_Comm_mem", make_comms)):
        try:
            live = reader(symbol)
        except ValueError:
            continue
        before = live()
        handles = make(16)
        during = live()
        for h in handles:
            h.Free()
        if during - before == 16 and live() == before:
            pools[key] = live
    return pools


POOLS = _mpich_pools()

if os.environ.get("LEAKY") == "1":
    def scatterDOFArray(self, a):
        """main f392a159 (proteus/mprans/VOF.py:980 there): one ghosted vector per id(a), over a itself."""
        if self.comm.size() == 1:
            return
        cache = self.__dict__.setdefault("_dofVecsById", {})
        par = cache.get(id(a))
        if par is None:
            dofMap = self.u[0].femSpace.dofMap
            rank = self.comm.rank()
            nOwned = dofMap.dof_offsets_subdomain_owned[rank + 1] - dofMap.dof_offsets_subdomain_owned[rank]
            par = cache[id(a)] = ParVec_petsc4py(
                a, 1, nOwned, dofMap.nDOF_all_processes, dofMap.nDOF_subdomain - nOwned, dofMap.subdomain2global)
        par.scatter_forward_insert()
    Transport.OneLevelTransport.scatterDOFArray = scatterDOFArray

_out = {}
_postStep = NS.NS_base.postStep
def postStep(self, model):
    result = _postStep(self, model)
    if model is self.modelList[-1]:
        if not _out:
            _out["f"] = open("growth.%d.csv" % Comm.get().rank(), "w")
            _out["f"].write("step,t,req,comm,parvec\n")
            _out["n"] = 0
        _out["n"] += 1
        parvecs = sum(1 for o in gc.get_objects() if isinstance(o, ParVec_petsc4py))
        _out["f"].write("%d,%.9g,%d,%d,%d\n" % (_out["n"], model.stepController.t_model,
                                                POOLS["req"]() if "req" in POOLS else -1,
                                                POOLS["comm"]() if "comm" in POOLS else -1, parvecs))
        _out["f"].flush()
    return result
NS.NS_base.postStep = postStep

sys.argv = ['parun', '--TwoPhaseFlow', 'damBreakLevelSet.py', '-l1', '-C',
            'final_time=%s dt_output=%s he=0.1' % (os.environ.get("FT", "0.5"), os.environ.get("DT_OUTPUT", "0.05"))]
runpy.run_path(shutil.which('parun'), run_name='__main__')
