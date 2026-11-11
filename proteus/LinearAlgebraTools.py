"""
Tools for n-dimensional linear algebra

Vectors are just numpy arrays, as are dense matrices. Sparse matrices
are CSR matrices. Parallel vector and matrix are built on top of those
representations using PETSc.

.. inheritance-diagram:: proteus.LinearAlgebraTools
   :parts: 1
"""
import numpy
from proteus import cmeshTools
from proteus.csparsity import CSR_INDEX_DTYPE
import math
import sys
from mpi4py import MPI
from . import superluWrappers
from . import Comm
from .Comm import globalSum, globalMax
from .superluWrappers import *
from .Profiling import logEvent
from .petsc_optional import PETSc as p4pyPETSc, HAVE_PETSC4PY

# PETSc Matrix Functions

def _petsc_view(obj, filename):
    """Saves petsc object to disk using a PETSc binary viewer.

    Parameters
    ----------
    obj : PETSc obj
        PETSc4py object to be saved (e.g. vector, matrix, etc)
    filename : str
        String with PETSc filename
    """
    viewer = p4pyPETSc.Viewer().createBinary(filename, 'w')
    viewer(obj)
    viewer2 = p4pyPETSc.Viewer().createASCII(filename+".m", 'w')
    viewer2.pushFormat(1)
    viewer2(obj)
    viewer2.popFormat()

def petsc_load_matrix(filename):
    """ This function loads a PETSc matrix from a binary format.
    (Eg. what is saved using the petsc_view function).

    Parameters
    ----------
    filename : str
        This is the name of the binary with the file stored.

    Returns
    -------
    matrix : petsc4py matrix
        The matrix that is stored in the binary file.
    """
    try:
        viewer = p4pyPETSc.Viewer().createBinary(filename,'r')
        output = p4pyPETSc.Mat().load(viewer)
    except:
        logEvent("Either you've entered an invalid file name or your object is not a matrix (try petsc_load_vector).")
        output = None
    return output

def petsc_load_vector(filename):
    """ This function loads a PETSc vector from a binary format.
    (Eg. what is saved using the petsc_view function).

    Parameters
    ----------
    filename : str
        This is the name of the binary with the file stored.

    Returns
    -------
    matrix : petsc4py vector
        The matrix that is stored in the binary file.
    """
    try:
        viewer = p4pyPETSc.Viewer().createBinary(filename,'r')
        output = p4pyPETSc.Vec().load(viewer)
    except:
        logEvent("Either you've entered an invalid file name or your object is not a vector (try petsc_load_matrix).")
        output = None
    return output

def petsc_load_IS(filename):
    """ This function loads a PETSc index-set from a binary format.
    (Eg. what is saved using the petsc_view function).

    Parameters
    ----------
    filename : str
        This is the name of the binary with the file stored.

    Returns
    -------
    matrix : petsc4py IS
        The index-set that is stored in the binary file.
    """
    try:
        viewer = p4pyPETSc.Viewer().createBinary(filename,'r')
        output = p4pyPETSc.IS().load(viewer)
    except:
        logEvent("Either you've entered an invalid file name or your object is not an index set.")
        output = None
    return output

def csr_2_petsc(size,csr):
    """ Create an petsc4py matrix from size and CSR information.

    Parameters
    ----------
    size : tuple
        A 2-tuple with the number of matrix rows and columns.
    csr : tuple
        A 3-tuple with the sparse matrix csr information.

    Returns
    --------
    matrix : PETSc4py aij matrix
    """
    mat = p4pyPETSc.Mat().create()
    mat.setSizes(size = size)
    mat.setType('aij')
    mat.setUp()
    mat.assemblyBegin()
    mat.setValuesCSR(csr[0],csr[1],csr[2])
    mat.assemblyEnd()
    return mat

def _pythonCSR_2_dense(rowptr,colptr,data,nr,nc,output=False):
    """ Takes python CSR datatypes and makes a dense matrix """
    dense_matrix = numpy.zeros(shape = (nr,nc), dtype='float')
    for idx in range(len(rowptr)-1):
        row_vals = data[rowptr[idx]:rowptr[idx+1]]
        for val_idx,j in enumerate(colptr[rowptr[idx]:rowptr[idx+1]]):
            dense_matrix[idx][j] = row_vals[val_idx]
    if output is not False:
        numpy.save(output,dense_matrix)
    return dense_matrix

def superlu_get_rank(sparse_matrix):
    """ Returns the rank of a superluWrapper sparse matrix.

    Parameters
    ----------
    sparse_matrix : :class:`proteus.superluWrappers.SparseMatrix`

    Returns
    -------
    matrix_rank : int
        The rank of the sparse_matrix

    Notes
    -----
    This function is a tool for debugging and should only be used
    for small matrices.
    """
    A = superlu_sparse_2_dense(sparse_matrix)
    return numpy.linalg.matrix_rank(A)

def petsc4py_get_rank(sparse_matrix):
    """ Returns the rank of a superluWrapper sparse matrix.

    Parameters
    ----------
    sparse_matrix : :class:`p4pyPETSc.Mat`

    Returns
    -------
    matrix_rank : int
        The rank of the sparse_matrix

    Notes
    -----
    This function is a debugging tool and should only be used
    for small matrices.
    """
    A = petsc4py_sparse_2_dense(sparse_matrix)
    return numpy.linalg.matrix_rank(A)

def superlu_has_pressure_null_space(sparse_matrix):
    """
    Checks whether a superluWrapper sparse matrix has a constant
    pressure null space.

    Parameters
    ----------
    sparse_matrix : :class:`proteus.superluWrappers.SparseMatrix`

    Returns
    -------
    does : bool
       Boolean variable indicating whether the pressure term
       creates a null space.

    Notes
    -----
    Assumes interwoven dof.
    This function was written mainly for debugging purposes and may be
    slow for large matrices.
    """
    A = superlu_2_petsc4py(sparse_matrix)
    return petsc4py_mat_has_pressure_null_space(A)

def petsc4py_mat_has_pressure_null_space(A):
    """
    Checks whether a PETSc4Py sparse matrix has a constant
    pressure null space.

    Parameters
    ----------
    A : :class:`p4pyPETSc.Mat`

    Returns
    -------
    does : bool
       Boolean variable indicating whether the pressure term
       creates a null space.

    Notes
    -----
    Assumes interwoven dof.
    This function was written mainly for debugging purposes and may be
    slow for large matrices.
    """
    x = numpy.zeros(A.getSize()[1])
    y = numpy.zeros(A.getSize()[1])
    x[::3] = 1
    x_petsc = p4pyPETSc.Vec().createWithArray(x)
    y_petsc = p4pyPETSc.Vec().createWithArray(y)
    A.mult(x_petsc,y_petsc)
    if y_petsc.norm() < 1e-15:
        return True
    else:
        return False

def superlu_sparse_2_dense(sparse_matrix,output=False):
    """ Converts a sparse superluWrapper into a dense matrix.

    Parameters
    ----------
    sparse_matrix :
    output : str
        Out file name to store the matrix.

    Returns
    -------
    dense_matrix : numpy array
        A numpy array storing the dense matrix.

    Notes
    -----
    This function should not be used for large matrices.
    """
    rowptr = sparse_matrix.getCSRrepresentation()[0]
    colptr = sparse_matrix.getCSRrepresentation()[1]
    data   = sparse_matrix.getCSRrepresentation()[2]
    nr     = sparse_matrix.shape[0]
    nc     = sparse_matrix.shape[1]
    return _pythonCSR_2_dense(rowptr,colptr,data,nr,nc,output)

def petsc4py_sparse_2_dense(sparse_matrix,output=False):
    """ Converts a PETSc4Py matrix to a dense numpyarray.

    Parameters
    ----------
    sparse_matrix : PETSc4py matrix
    output : str
        Output file name to store the matrix.

    Returns
    -------
    dense_matrix : numpy array
        A numpy array with the dense matrix.

    Notes
    -----
    This function is very inefficient for large matrices.
    """
    rowptr = sparse_matrix.getValuesCSR()[0]
    colptr = sparse_matrix.getValuesCSR()[1]
    data   = sparse_matrix.getValuesCSR()[2]
    nr     = sparse_matrix.getSize()[0]
    nc     = sparse_matrix.getSize()[1]
    return _pythonCSR_2_dense(rowptr,colptr,data,nr,nc,output)

def superlu_2_petsc4py(sparse_superlu):
    """ Copy a sparse superlu matrix to a sparse petsc4py matrix

    Parameters
    ----------
    sparse_superlu : :class:`proteus.superluWrappers.SparseMatrix`

    Returns
    -------
    sparse_matrix : :class: `p4pyPETSc.Mat`
    """
    comm = Comm.get()

    if comm.size() > 1:
        rowptr,colind,nzval = sparse_superlu.getCSRrepresentation()
        A_petsc4py = ParMat_petsc4py.create_ParMat_from_OperatorConstructor(sparse_superlu)

    else:
        rowptr, colind, nzval = sparse_superlu.getCSRrepresentation()
        A_rowptr = rowptr.copy()
        A_colind = colind.copy()
        A_nzval  = nzval.copy()
        nr       = sparse_superlu.shape[0]
        nc       = sparse_superlu.shape[1]
        A_petsc4py = p4pyPETSc.Mat().createAIJWithArrays((nr,nc),
                                                         (A_rowptr,
                                                          A_colind,
                                                          A_nzval))
    return A_petsc4py

def petsc_create_diagonal_inv_matrix(sparse_petsc):
    """ Create an inverse diagonal petsc4py matrix from input matrix.

    Parameters
    ----------
    sparse_petsc : :class:`p4pyPETSc.Mat`

    Returns
    -------
    sparse_matrix : :class:`p4pyPETSc.Mat`
    """
    diag_inv = p4pyPETSc.Mat().create()
    diag_inv.setSizes(sparse_petsc.getSizes())
    diag_inv.setType('aij')
    diag_inv.setUp()
    oneByD = sparse_petsc.getDiagonal()
    oneByD.reciprocal()
    diag_inv.setDiagonal(oneByD)
    return diag_inv

def dense_numpy_2_petsc4py(dense_numpy, eps = 1.e-12):
    """ Create a sparse petsc4py matrix from a dense numpy matrix.

    Note - This routine has been built mainly to support testing.
    It would be rare for this routine to be useful for most applications.

    Parameters
    ----------
    dense_numpy :
    eps : float
        Tolerance for non-zero values.

    Returns
    -------
    sparse_matrix : PETSc4py matrix
    """
    vals = []
    colptr = []
    rowptr = [0]
    rowptr_track = 0
    for i,row in enumerate(dense_numpy):
        for j,val in enumerate(row):
            if abs(val) > eps:
                vals.append(val)
                colptr.append(j)
                rowptr_track += 1
        rowptr.append(rowptr_track)
    return p4pyPETSc.Mat().createAIJ(size=dense_numpy.shape,
                                     csr = (rowptr, colptr, vals))
def csr_2_petsc_mpiaij(size,csr):
    """ Create an MPIaij petsc4py matrix from size and CSR information.

    Parameters
    ----------
    size : tuple
        Two entires: (num_rows, num_cols)
    csr : tuple
        (row_idx, col_idx, vals)

    Returns
    --------
    matrix : PETSc4py MPIaij matrix
    """
    mat = p4pyPETSc.Mat().create()
    mat.setSizes(size = size)
    mat.setType('mpiaij')
    mat.setUp()
    mat.assemblyBegin()
    mat.setValuesCSR(csr[0],csr[1],csr[2])
    mat.assemblyEnd()
    return mat

def split_PETSc_Mat(mat):
    """ Decompose a PETSc matrix into a symmetric and skew-symmetric
        matrix

    Parameters
    ----------
    mat : :class: `PETSc4py Matrix`

    Returns
    --------
    H : :class: `PETSc4py Matrix`
        Symmetric (or Hermitian) component of mat
    S : :class: `PETSc4py Matrix`
        Skew-Symmetric (or skew-Hermitian) component of mat
    """
    H = mat.copy()
    H.zeroEntries()
    H.axpy(1.0,mat)
    H.axpy(1.0,mat.transpose())
    H.scale(0.5)
    S = mat.copy()
    S.zeroEntries()
    S.axpy(1.0,mat)
    S.aypx(-1.0,mat.transpose())
    S.scale(0.5)
    return H, S

class ParVec_petsc4py(p4pyPETSc.Vec):
    """
    Parallel vector using petsc4py's wrappers for PETSc

    Parameters
    ----------
    array : numpy_array
            A numpy array with size equal to the number of locally
            owned unknowns plus the number of local ghost cells.
    bs : int
         Block size.
    n : int
        The number of locally owned unknowns
    N : int
        The number of unknowns in the global system
    nghosts : int
              The number of ghost nodes for the process.
    subdomain2global : numpy array
                       Map from the process unknowns to the global
                       uknowns.
    blockVecType : str
    ghosts : numpy array
             A numpy array with the local process uknowns that are
             ghost nodes.
    proteus2petsc_subdomain : numpy array
             A numpy array that serves as a map from the proteus
             uknown ordering to the petsc uknown ordering
    petsc2proteus_subdomain : numpy array
            A numpy array that serves as a map from the petsc uknown
            ordering to the proteus unknown ordering
    """
    dirichlet_nonauthority = None  # local dofs dropped before the reverse add, see strongDirichletNonAuthorityDofs
    def __init__(self,array=None,bs=None,n=None,N=None,nghosts=None,subdomain2global=None,blockVecType="simple",ghosts=None,
                                                 proteus2petsc_subdomain=None,
                                                 petsc2proteus_subdomain=None):
        p4pyPETSc.Vec.__init__(self)
        if array is None:
            return#when duplicating for petsc usage
        self.proteus2petsc_subdomain=proteus2petsc_subdomain
        self.petsc2proteus_subdomain=petsc2proteus_subdomain
        blockSize = max(1,bs)
        self.dim_proc = n*blockSize
        self.nghosts = nghosts
        self.blockVecType = blockVecType
        assert self.blockVecType == "simple", "petsc4py wrappers require self.blockVecType=simple"
        self.proteus_array = array
        # Remember the layout so a sibling vector can be built over a different
        # array (see duplicate_with_array). subdomain2global is kept here in its
        # original, nodal form because the bs > 1 branch below replaces
        # self.subdomain2global with a block-expanded copy.
        self.bs = bs
        self.n = n
        self.N = N
        self.nodal_subdomain2global = subdomain2global
        self.ghosts_array = ghosts
        if nghosts is None:
            if blockVecType == "simple":
                self.createWithArray(array,size=(blockSize*n,blockSize*N),bsize=1)
            else:
                self.createWithArray(array,size=(blockSize*n,blockSize*N),bsize=blockSize)
            self.subdomain2global=subdomain2global
            self.petsc_l2g = None
            self.setUp()
        else:
            assert nghosts >= 0, "The number of ghostnodes must be non-negative"
            assert subdomain2global.shape[0] == (n+nghosts), ("The subdomain2global map is the wrong length n=%i,nghosts=%i,shape=%i \n" % (n,n+nghosts,subdomain2global.shape[0]))
            assert len(array.flat) == (n+nghosts)*blockSize, "%i  != (%i+%i)*%i \n" % (len(array.flat),  n,nghosts,blockSize)
            if blockVecType == "simple":
                if ghosts is None:
                    ghosts = numpy.zeros((blockSize*nghosts),'i')
                    for j in range(blockSize):
                        ghosts[j::blockSize]=subdomain2global[n:]*blockSize+j
                self.createGhostWithArray(ghosts,array,size=(blockSize*n,blockSize*N),bsize=1)
                if blockSize > 1: #have to build in block dofs
                    subdomain2globalTotal = numpy.zeros((blockSize*subdomain2global.shape[0],),cmeshTools.INDEX_DTYPE)
                    for j in range(blockSize):
                        subdomain2globalTotal[j::blockSize]=subdomain2global*blockSize+j
                    self.subdomain2global=subdomain2globalTotal
                else:
                    self.subdomain2global=subdomain2global
            else:
                #TODO need to debug
                ghosts = subdomain2global[n:]
                self.createGhostWithArray(ghosts,array,size=(blockSize*n,blockSize*N),bsize=blockSize)
                self.subdomain2global = subdomain2global
            self.setUp()
            #self.petsc_l2g = p4pyPETSc.LGMap()
            #self.petsc_l2g.create(self.subdomain2global)
            #self.setLGMap(self.petsc_l2g)
        self.setFromOptions()
    def duplicate_with_array(self,array):
        """A new ParVec over `array`, carrying this vector's parallel layout.

        petsc4py's own duplicate() allocates PETSc-owned storage, which is no
        use when the caller already has a numpy array it needs the solvers to
        read and write in place -- NI's per-level work vectors, for instance.
        """
        assert array.size == self.proteus_array.size, (
            "array has %i entries but this vector's layout expects %i"
            % (array.size, self.proteus_array.size))
        return ParVec_petsc4py(array,
                               self.bs,
                               self.n,
                               self.N,
                               self.nghosts,
                               self.nodal_subdomain2global,
                               self.blockVecType,
                               self.ghosts_array,
                               self.proteus2petsc_subdomain,
                               self.petsc2proteus_subdomain)
    def scatter_forward_insert(self):
        if self.proteus2petsc_subdomain is not None:
            self.proteus_array[:] = self.proteus_array[self.petsc2proteus_subdomain]
        self.ghostUpdateBegin(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        self.ghostUpdateEnd(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        if self.proteus2petsc_subdomain is not None:
            self.proteus_array[:] = self.proteus_array[self.proteus2petsc_subdomain]
    def scatter_reverse_add(self):
        if self.dirichlet_nonauthority is not None:
            # strong Dirichlet dofs: this rank's copy of the row is not the authority's (see
            # strongDirichletNonAuthorityDofs); drop it so the sum over the copies is one u - g
            self.proteus_array.reshape(-1)[self.dirichlet_nonauthority] = 0.0
        if self.proteus2petsc_subdomain is not None:
            self.proteus_array[:] = self.proteus_array[self.petsc2proteus_subdomain]
        self.ghostUpdateBegin(p4pyPETSc.InsertMode.ADD_VALUES,p4pyPETSc.ScatterMode.REVERSE)
        self.ghostUpdateEnd(p4pyPETSc.InsertMode.ADD_VALUES,p4pyPETSc.ScatterMode.REVERSE)
        if self.proteus2petsc_subdomain is not None:
            self.proteus_array[:] = self.proteus_array[self.proteus2petsc_subdomain]

    def save(self, filename):
        """Saves to disk using a PETSc binary viewer."""
        _petsc_view(self, filename)

# Parallel vectors and transfer matrices over mpi4py alone
#
# Proteus's own solvers need from PETSc only a ghosted vector (to refresh the
# ghost entries a subdomain matrix reads) and, for multigrid, the transfer
# operators applied to owned rows. Both are a halo exchange plus local sparse
# products, which is what follows. They are used when petsc4py cannot be
# imported; see petsc_optional.

class GhostExchange(object):
    """The communication pattern of a ghosted layout: who sends which owned entry to whom.

    Entries are numbered globally, each rank owning the contiguous range
    [offsets[rank], offsets[rank+1]) -- PETSc's layout, and the one partitioning
    produces. ``ghostGlobal`` lists the global indices whose values this rank
    needs but does not own. Building the plan is collective (each rank tells
    the owners what it needs, once); after that, forward() gives the ghosts
    their owners' values and reverse_add() adds ghost contributions into the
    owners -- PETSc's ghostUpdate INSERT/FORWARD and ADD/REVERSE.
    """
    def __init__(self,mpi,offsets,ghostGlobal):
        from mpi4py import MPI
        self.mpi = mpi
        rank,size = mpi.Get_rank(),mpi.Get_size()
        offsets = numpy.asarray(offsets,dtype=numpy.int64)
        ghostGlobal = numpy.asarray(ghostGlobal,dtype=numpy.int64).ravel()
        self.nGhosts = ghostGlobal.size
        owners = numpy.searchsorted(offsets,ghostGlobal,side='right')-1
        if self.nGhosts and (owners.min() < 0 or owners.max() >= size or (owners == rank).any()):
            raise ValueError("ghost entries must be owned by another rank and lie in [0,%i)"
                             % offsets[-1])
        #ghost values arrive grouped by owning rank; recvOrder puts them back
        self.recvOrder = numpy.argsort(owners,kind='stable')
        self.recvCounts = numpy.bincount(owners,minlength=size).astype(numpy.int64)
        self.sendCounts = numpy.asarray(mpi.alltoall(self.recvCounts.tolist()),dtype=numpy.int64)
        self.recvDispl = numpy.concatenate([[0],numpy.cumsum(self.recvCounts)[:-1]])
        self.sendDispl = numpy.concatenate([[0],numpy.cumsum(self.sendCounts)[:-1]])
        requested = numpy.ascontiguousarray(ghostGlobal[self.recvOrder])
        incoming = numpy.empty((int(self.sendCounts.sum()),),dtype=numpy.int64)
        mpi.Alltoallv([requested,(self.recvCounts,self.recvDispl),MPI.INT64_T],
                      [incoming,(self.sendCounts,self.sendDispl),MPI.INT64_T])
        self.sendLocal = incoming - offsets[rank]
        nOwned = int(offsets[rank+1]-offsets[rank])
        if self.sendLocal.size and (self.sendLocal.min() < 0 or self.sendLocal.max() >= nOwned):
            raise ValueError("a rank asked for an entry this rank does not own")

    def forward(self,owned):
        """The ghost values, in ghostGlobal's order, taken from their owners."""
        from mpi4py import MPI
        sendbuf = numpy.ascontiguousarray(owned[self.sendLocal],dtype=numpy.float64)
        recvbuf = numpy.empty((self.nGhosts,),dtype=numpy.float64)
        self.mpi.Alltoallv([sendbuf,(self.sendCounts,self.sendDispl),MPI.DOUBLE],
                           [recvbuf,(self.recvCounts,self.recvDispl),MPI.DOUBLE])
        ghosts = numpy.empty_like(recvbuf)
        ghosts[self.recvOrder] = recvbuf
        return ghosts

    def reverse_add(self,ghosts,owned):
        """Add each ghost value into its owner's entry of ``owned`` (in place)."""
        from mpi4py import MPI
        sendbuf = numpy.ascontiguousarray(numpy.asarray(ghosts,dtype=numpy.float64)[self.recvOrder])
        recvbuf = numpy.empty((self.sendLocal.size,),dtype=numpy.float64)
        self.mpi.Alltoallv([sendbuf,(self.recvCounts,self.recvDispl),MPI.DOUBLE],
                           [recvbuf,(self.sendCounts,self.sendDispl),MPI.DOUBLE])
        numpy.add.at(owned,self.sendLocal,recvbuf)


def _ownershipOffsets(mpi,nOwned):
    """Global offsets of a contiguous layout from each rank's owned count."""
    counts = mpi.allgather(int(nOwned))
    return numpy.concatenate([[0],numpy.cumsum(counts)]).astype(numpy.int64)


class ParVec_mpi4py(object):
    """A ghosted parallel vector over a numpy array, using mpi4py for the halo exchange.

    Takes the arguments ParVec_petsc4py takes and lays the array out the same
    way: the first n*bs entries are owned, globally numbered from this rank's
    offset, and the rest are ghosts whose global numbers come from
    subdomain2global (or ``ghosts``). It provides what proteus's own solvers
    use: scatter_forward_insert, scatter_reverse_add, duplicate_with_array and
    the layout attributes. It is not a PETSc Vec, so a PETSc solver cannot use
    it -- which is the point: it is what a run without petsc4py gets.
    """
    def __init__(self,array=None,bs=None,n=None,N=None,nghosts=None,subdomain2global=None,
                 blockVecType="simple",ghosts=None,
                 proteus2petsc_subdomain=None,petsc2proteus_subdomain=None,
                 _exchange=None):
        from . import Comm
        self.proteus2petsc_subdomain = proteus2petsc_subdomain
        self.petsc2proteus_subdomain = petsc2proteus_subdomain
        blockSize = max(1,bs)
        self.dim_proc = n*blockSize
        self.nghosts = nghosts
        self.blockVecType = blockVecType
        self.proteus_array = array
        self.bs = bs
        self.n = n
        self.N = N
        self.nodal_subdomain2global = subdomain2global
        self.ghosts_array = ghosts
        if blockSize > 1 and subdomain2global is not None:
            total = numpy.zeros((blockSize*subdomain2global.shape[0],),'i')
            for j in range(blockSize):
                total[j::blockSize] = subdomain2global*blockSize+j
            self.subdomain2global = total
        else:
            self.subdomain2global = subdomain2global
        if nghosts is None:
            self.exchange = None
            return
        assert nghosts >= 0, "The number of ghostnodes must be non-negative"
        assert subdomain2global.shape[0] == (n+nghosts), (
            "The subdomain2global map is the wrong length n=%i,nghosts=%i,shape=%i"
            % (n,n+nghosts,subdomain2global.shape[0]))
        assert len(array.flat) == (n+nghosts)*blockSize, "%i  != (%i+%i)*%i" % (
            len(array.flat),n,nghosts,blockSize)
        if _exchange is None:
            if ghosts is None:
                ghosts = numpy.zeros((blockSize*nghosts),'i')
                for j in range(blockSize):
                    ghosts[j::blockSize] = subdomain2global[n:]*blockSize+j
            mpi = Comm.get().comm.tompi4py()
            _exchange = GhostExchange(mpi,_ownershipOffsets(mpi,self.dim_proc),ghosts)
        self.exchange = _exchange

    def duplicate_with_array(self,array):
        """A new ParVec_mpi4py over ``array`` with this one's layout; no communication."""
        assert array.size == self.proteus_array.size, (
            "array has %i entries but this vector's layout expects %i"
            % (array.size,self.proteus_array.size))
        return ParVec_mpi4py(array,self.bs,self.n,self.N,self.nghosts,
                             self.nodal_subdomain2global,self.blockVecType,self.ghosts_array,
                             self.proteus2petsc_subdomain,self.petsc2proteus_subdomain,
                             _exchange=self.exchange)

    def getArray(self):
        """The owned entries, as a ghosted PETSc Vec's getArray() returns them."""
        return self.proteus_array.reshape(-1)[:self.dim_proc]

    def _flat(self):
        flat = self.proteus_array.reshape(-1)
        #an empty array (a rank that owns nothing) shares no memory with anything
        assert flat.size == 0 or numpy.shares_memory(flat,self.proteus_array), "the vector's array must be contiguous"
        return flat

    def scatter_forward_insert(self):
        if self.exchange is None:
            return
        flat = self._flat()
        if self.proteus2petsc_subdomain is not None:
            flat[:] = flat[self.petsc2proteus_subdomain]
        flat[self.dim_proc:] = self.exchange.forward(flat[:self.dim_proc])
        if self.proteus2petsc_subdomain is not None:
            flat[:] = flat[self.proteus2petsc_subdomain]

    def scatter_reverse_add(self):
        if self.exchange is None:
            return
        flat = self._flat()
        if self.proteus2petsc_subdomain is not None:
            flat[:] = flat[self.petsc2proteus_subdomain]
        self.exchange.reverse_add(flat[self.dim_proc:],flat[:self.dim_proc])
        if self.proteus2petsc_subdomain is not None:
            flat[:] = flat[self.proteus2petsc_subdomain]


class ParMat_mpi4py(object):
    """The parallel layout of a subdomain matrix, for solvers that do not need PETSc.

    Takes ParMat_petsc4py's arguments but assembles nothing: proteus's own
    solvers apply the subdomain matrix itself and need from the parallel
    matrix only which rows are owned and how they are numbered globally --
    the redundant coarse solve (RedundantLU) gathers those rows.
    """
    def __init__(self,ghosted_csr_mat=None,par_bs=None,par_n=None,par_N=None,par_nghost=None,
                 subdomain2global=None,blockVecType="simple",pde=None,par_nc=None,par_Nc=None,
                 proteus_jacobian=None,nzval_proteus2petsc=None):
        self.ghosted_csr_mat = ghosted_csr_mat
        self.blockSize = max(1,par_bs)
        self.par_n = par_n
        self.par_N = par_N
        self.par_nghost = par_nghost
        self.subdomain2global = subdomain2global
        self.pde = pde
        #set for mixed spaces, whose parallel numbering is a permutation of
        #proteus's; RedundantLU supports only the plain layout
        self.nzval_proteus2petsc = nzval_proteus2petsc


class DistVec(object):
    """A rank's owned entries of a distributed vector: the Vec a DistMat creates."""
    def __init__(self,n):
        self.array = numpy.zeros((n,),'d')

    def getArray(self):
        return self.array

    def set(self,value):
        self.array[:] = value


class DistMat(object):
    """A row-distributed sparse matrix over mpi4py, for the multilevel transfer operators.

    It provides the part of a petsc4py Mat the transfer operators are built and
    applied with -- setSizes-style construction, setValues on owned rows in
    global numbers, assemblyBegin/End, createVecLeft/Right, mult and
    multTranspose -- so the same constructors produce either. After assembly
    the owned rows are a local CSR matrix whose columns are this rank's owned
    columns followed by the off-rank ones it reads; mult fetches those with a
    GhostExchange and multTranspose sends its contributions to them back to
    their owners and adds them there, as PETSc's MPIAIJ does.
    """
    def __init__(self,mpi,sizes):
        (self.n,self.N),(self.m,self.M) = sizes
        self.mpi = mpi
        self.rank = mpi.Get_rank()
        self.rowOffsets = _ownershipOffsets(mpi,self.n)
        self.colOffsets = _ownershipOffsets(mpi,self.m)
        assert self.rowOffsets[-1] == self.N and self.colOffsets[-1] == self.M, (
            "local sizes do not add up to the global ones")
        self._rows = {}
        self.csr = None

    def getSizes(self):
        return ((self.n,self.N),(self.m,self.M))

    def setValues(self,rows,cols,values):
        cols = [int(c) for c in cols]
        values = numpy.asarray(values,'d').reshape(len(rows),len(cols))
        lo = int(self.rowOffsets[self.rank])
        for i,r in enumerate(rows):
            r = int(r)
            assert lo <= r < lo+self.n, "row %i is not owned by this rank" % r
            entries = self._rows.setdefault(r-lo,{})
            for c,v in zip(cols,values[i]):
                entries[c] = v

    def assemblyBegin(self):
        pass

    def assemblyEnd(self):
        import scipy.sparse
        lo = int(self.colOffsets[self.rank])
        rowsOf,colsOf,vals = [],[],[]
        for r,entries in self._rows.items():
            for c,v in entries.items():
                rowsOf.append(r); colsOf.append(c); vals.append(v)
        colsOf = numpy.asarray(colsOf,dtype=numpy.int64)
        local = (colsOf >= lo) & (colsOf < lo+self.m)
        offRank = numpy.unique(colsOf[~local])
        self.exchange = GhostExchange(self.mpi,self.colOffsets,offRank)
        mapped = numpy.where(local,colsOf-lo,self.m+numpy.searchsorted(offRank,colsOf))
        self.csr = scipy.sparse.csr_matrix((numpy.asarray(vals,'d'),
                                            (numpy.asarray(rowsOf,dtype=numpy.int64),mapped)),
                                           shape=(self.n,self.m+offRank.size))
        self.csrT = self.csr.T.tocsr()
        self._rows = None

    def createVecLeft(self):
        return DistVec(self.n)

    def createVecRight(self):
        return DistVec(self.m)

    def mult(self,x,y):
        owned = x.getArray()
        extended = numpy.concatenate([owned,self.exchange.forward(owned)])
        y.getArray()[:] = self.csr.dot(extended)

    def multTranspose(self,y,x):
        extended = self.csrT.dot(y.getArray())
        result = x.getArray()
        result[:] = extended[:self.m]
        self.exchange.reverse_add(extended[self.m:],result)


def newTransferMat(comm,sizes,d_nnz,o_nnz):
    """An empty distributed matrix for a transfer operator: PETSc's, or a DistMat without petsc4py."""
    if not HAVE_PETSC4PY:
        return DistMat(comm.comm.tompi4py(),sizes)
    mat = p4pyPETSc.Mat().create(comm.comm.tompi4py())
    mat.setType('aij')
    mat.setSizes(sizes)
    mat.setPreallocationNNZ((d_nnz,o_nnz))
    mat.setOption(p4pyPETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)
    return mat


class ParInfo_petsc4py(object):
    """
    ARB - this class is experimental.  My idea is to store the
    information need to constructor parallel vectors and matrices
    here as static class values.  Then ParVec and ParMat can
    use these values to create parallel objects later.
    """

    def __init__(self):
        self.par_bs = None
        self.par_n = None
        self.par_n_lst = None
        self.par_N = None
        self.par_nghost = None
        self.par_nghost_lst = None
        self.petsc_subdomain2global_petsc = None
        self.subdomain2global = None
        self.proteus2petsc_subdomain = None
        self.petsc2proteus_subdomain = None
        self.nzval_proteus2petsc = None
        self.dim = None
        self.mixed = False

    def print_info(cls):
        from . import Comm
        comm = Comm.get()
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_bs = ' + repr(cls.par_bs))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_n = ' + repr(cls.par_n))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_n_lst = ' + repr(cls.par_n_lst))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_N = ' + repr(cls.par_N))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_nghost = ' + repr(cls.par_nghost))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' par_nghost_lst = ' + repr(cls.par_nghost_lst))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' petsc_subdomain2global_petsc = ' + repr(cls.petsc_subdomain2global_petsc))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' subdomain2global = ' + repr(cls.subdomain2global))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' proteus2petsc_subdomain = ' + repr(cls.proteus2petsc_subdomain))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' petsc2proteus_subomdain = ' + repr(cls.petsc2proteus_subdomain))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' dim = ' + repr(cls.dim))
        logEvent('comm.rank() = ' + repr(comm.rank()) + ' nzval_proteus2petsc = ' + repr(cls.nzval_proteus2petsc))

class ParMat_petsc4py(p4pyPETSc.Mat):
    """  Parallel matrix based on petsc4py's wrappers for PETSc.

    Parameters
    ----------
    ghosted_csr_mat : :class:`proteus.superluWrappers.SparseMatrix`
        Primary CSR information for the ParMat.
    par_bs : int
        The block size.
    par_n : int
        The number of locally owned unknowns.
    par_N : int
        The number of global unknowns.
    par_nghost : int
        The number of locally owned ghost unknowns.
    subdomain2global : :class:`numpy.ndarray`
        A map from the local unknown to the global unknown.
    blockVecType : str
    pde : :class:`proteus.Transport.OneLevelTransport`
        The Transport class defining the problem.
    par_nc : int
    par_Nc : int
    proteus_jacobian : :class:`proteus.superluWrappers.SparseMatrix`
        Jacobian generated by Transport class's initializeJacobian.
    nzval_proteus2petsc : :class:`numpy.ndarray`
        Array with index permutations for mapping between
        proteus and petsc degrees of freedom.
    """
    def __init__(self,
                 ghosted_csr_mat=None,
                 par_bs=None,
                 par_n=None,
                 par_N=None,
                 par_nghost=None,
                 subdomain2global=None,
                 blockVecType="simple",
                 pde=None,
                 par_nc=None,
                 par_Nc=None,
                 proteus_jacobian=None,
                 nzval_proteus2petsc=None):
        p4pyPETSc.Mat.__init__(self)
        if ghosted_csr_mat is None:
            return#when duplicating for petsc usage
        self.pde = pde
        if par_nc is None:
            par_nc = par_n
        if par_Nc is None:
            par_Nc = par_N
        self.proteus_jacobian=proteus_jacobian
        self.nzval_proteus2petsc = nzval_proteus2petsc
        self.ghosted_csr_mat=ghosted_csr_mat
        self.blockVecType = blockVecType
        assert self.blockVecType == "simple", "petsc4py wrappers require self.blockVecType=simple"
        self.create(p4pyPETSc.COMM_WORLD)
        self.blockSize = max(1,par_bs)
        if self.blockSize > 1 and blockVecType != "simple":
            ## \todo fix block aij in ParMat_petsc4py
            self.setType('mpibaij')
            self.setSizes([[self.blockSize*par_n,self.blockSize*par_N],[self.blockSize*par_nc,self.blockSize*par_Nc]],bsize=self.blockSize)
            self.setBlockSize(self.blockSize)
            self.subdomain2global = subdomain2global #no need to include extra block dofs?
        else:
            self.setType('aij')
            self.setSizes([[par_n*self.blockSize,par_N*self.blockSize],[par_nc*self.blockSize,par_Nc*self.blockSize]],bsize=1)
            if self.blockSize > 1: #have to build in block dofs
                subdomain2globalTotal = numpy.zeros((self.blockSize*subdomain2global.shape[0],),cmeshTools.INDEX_DTYPE)
                for j in range(self.blockSize):
                    subdomain2globalTotal[j::self.blockSize]=subdomain2global*self.blockSize+j
                self.subdomain2global=subdomain2globalTotal
            else:
                self.subdomain2global=subdomain2global
        from proteus import Comm
        comm = Comm.get()
        logEvent("ParMat_petsc4py comm.rank= %s blockSize = %s par_n= %s par_N=%s par_nghost=%s par_jacobian.getSizes()= %s "
                 % (comm.rank(),self.blockSize,par_n,par_N,par_nghost,self.getSizes()))
        self.csr_rep = ghosted_csr_mat.getCSRrepresentation()
        if self.proteus_jacobian is not None:
            self.proteus_csr_rep = self.proteus_jacobian.getCSRrepresentation()
        if self.blockSize > 1:
            blockOwned = self.blockSize*par_n
            self.csr_rep_local = ghosted_csr_mat.getSubMatCSRrepresentation(0,blockOwned)
        else:
            self.csr_rep_local = ghosted_csr_mat.getSubMatCSRrepresentation(0,par_n)
        self.petsc_l2g = p4pyPETSc.LGMap()
        self.petsc_l2g.create(self.subdomain2global)
        self.setUp()
        self.setLGMap(self.petsc_l2g)
        #
        self.colind_global = self.petsc_l2g.apply(self.csr_rep_local[1]) #prealloc needs global indices
        self.setPreallocationCSR([self.csr_rep_local[0],self.colind_global,self.csr_rep_local[2]])
        self.setFromOptions()


    @classmethod
    def create_ParMat_from_OperatorConstructor(cls,
                                               operator):
        """ Build a ParMat consistent with the problem from an Operator
        constructor matrix.

        Arguments
        ---------
        operator : :class:`proteus.superluWrappers.SparseMatrix`
            Matrix to be turned into a parallel petsc matrix.
        """
        par_bs = ParInfo_petsc4py.par_bs
        par_n = ParInfo_petsc4py.par_n
        par_N = ParInfo_petsc4py.par_N
        par_nghost = ParInfo_petsc4py.par_nghost
        petsc_subdomain2global_petsc = ParInfo_petsc4py.petsc_subdomain2global_petsc
        subdomain2global = ParInfo_petsc4py.subdomain2global
        petsc2proteus_subdomain = ParInfo_petsc4py.petsc2proteus_subdomain
        proteus2petsc_subdomain = ParInfo_petsc4py.proteus2petsc_subdomain
        dim = ParInfo_petsc4py.dim
        # ARB - this is largely copied from Transport.py,
        # a refactor should be done to elimate this duplication
        rowptr, colind, nzval = operator.getCSRrepresentation()

        rowptr_petsc = rowptr.copy()
        colind_petsc = colind.copy()
        nzval_petsc = nzval.copy()
        nzval_proteus2petsc = colind.copy()
        nzval_petsc2proteus = colind.copy()
        rowptr_petsc[0] = 0

        for i in range(par_n+par_nghost):
            start_proteus = rowptr[petsc2proteus_subdomain[i]]
            end_proteus = rowptr[petsc2proteus_subdomain[i]+1]
            nzrow = end_proteus - start_proteus
            rowptr_petsc[i+1] = rowptr_petsc[i] + nzrow
            start_petsc = rowptr_petsc[i]
            end_petsc = rowptr_petsc[i+1]
            petsc_cols_i = proteus2petsc_subdomain[colind[start_proteus:end_proteus]]
            j_sorted = petsc_cols_i.argsort()
            colind_petsc[start_petsc:end_petsc] = petsc_cols_i[j_sorted]
            nzval_petsc[start_petsc:end_petsc] = nzval[start_proteus:end_proteus][j_sorted]
            for j_petsc, j_proteus in zip(numpy.arange(start_petsc,end_petsc),
                                          numpy.arange(start_proteus,end_proteus)[j_sorted]):
                nzval_petsc2proteus[j_petsc] = j_proteus
                nzval_proteus2petsc[j_proteus] = j_petsc

        proteus_a = {}
        petsc_a = {}

        for i in range(dim):
            for j,k in zip(colind[rowptr[i]:rowptr[i+1]],list(range(rowptr[i],rowptr[i+1]))):
                proteus_a[i,j] = nzval[k]
                petsc_a[proteus2petsc_subdomain[i],proteus2petsc_subdomain[j]] = nzval[k]
        for i in range(dim):
            for j,k in zip(colind_petsc[rowptr_petsc[i]:rowptr_petsc[i+1]],list(range(rowptr_petsc[i],rowptr_petsc[i+1]))):
                nzval_petsc[k] = petsc_a[i,j]

        #additional stuff needed for petsc par mat

        petsc_jacobian = SparseMat(dim,dim,nzval_petsc.shape[0], nzval_petsc, colind_petsc, rowptr_petsc)
        return cls(petsc_jacobian,
                   par_bs,
                   par_n,
                   par_N,
                   par_nghost,
                   petsc_subdomain2global_petsc,
                   proteus_jacobian = operator,
                   nzval_proteus2petsc=nzval_proteus2petsc)

    def save(self, filename):
        """Saves to disk using a PETSc binary viewer. """
        _petsc_view(self, filename)

def Vec(n):
    """
    Build a vector of length n (using numpy)

    For example::

      >>> Vec(3)
      array([ 0.,  0.,  0.])

    """
    return numpy.zeros((n,),'d')


def Mat(m,n):
    """
    Build an m x n matrix (using numpy)

    For example::

      >>> Mat(2,3)
      array([[ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

    """
    return numpy.zeros((m,n),'d')


def SparseMatFromDict(nr,nc,aDict):
    """
    Build a nr x nc sparse matrix from a dictionary representation

    Rows with no entries are fine, anywhere -- first, last or between others. (The row pointer was
    built by stepping one row at a time, so a gap in the row numbers shifted every later row up and
    left the trailing pointers at zero. A global operator has no empty rows; a rank's piece of one does:
    its ghost rows whose coarse parents are not on the rank.)
    """
    from . import superluWrappers
    indeces = list(aDict.keys())
    indeces.sort()
    nnz     = len(indeces)
    nzval   = numpy.zeros((nnz,),'d')
    colind  = numpy.zeros((nnz,),CSR_INDEX_DTYPE)
    rows    = numpy.zeros((nnz,),CSR_INDEX_DTYPE)
    for k,ij in enumerate(indeces):
        nzval[k] = aDict[ij]
        rows[k] = ij[0]
        colind[k] = ij[1]
    assert nnz == 0 or (rows.min() >= 0 and rows.max() < nr), "an entry lies outside the %d rows" % nr
    rowptr  = numpy.zeros((nr+1,),CSR_INDEX_DTYPE)
    rowptr[1:] = numpy.cumsum(numpy.bincount(rows,minlength=nr))
    return (SparseMat(nr,nc,nnz,nzval,colind,rowptr),nzval)


def SparseMat(nr,nc,nnz,nzval,colind,rowptr):
    """ Build a nr x nc sparse matrix from the CSR data structures

    Parameters
    ----------
    nr : int
        The number of rows.
    nc : int
        The number of columns.
    nnz : int
        The number of non-zero matrix entries.
    nzval : numpy array
        Array with non-zero matrix entries.
    colind : numpy array of 32bit integers
        CSR column array.
    rowptr : numpy array of 32bit integers
        CSR row pointer.

    Returns
    -------
    sparse_matrix : :class:`proteus.superluWrappers.SparseMatrix`
        superlu sparse matrix in CSR format.

    Note
    ----
    colind and rowptr must be csparsity.CSR_INDEX_DTYPE arrays (the width of
    PetscInt). superluWrappers converts them to SuperLU's 32-bit int itself,
    and only if the two widths differ.
    """
    if (colind.dtype != CSR_INDEX_DTYPE or rowptr.dtype != CSR_INDEX_DTYPE):
        print('ERROR - colind and rowptr must be %s numpy arrays '
              '(csparsity.CSR_INDEX_DTYPE)' % CSR_INDEX_DTYPE)
        sys.exit(1)
    return superluWrappers.SparseMatrix(nr,nc,nnz,nzval,colind,rowptr)

class SparseMatShell(object):
    """ Build a parallel matrix shell from CSR data structures.

    Parameters
    ----------
    ghosted_csr_mat: :class: `proteus.superluWrappers.SparseMatrix`
    """
    def __init__(self,ghosted_csr_mat):
        self.ghosted_csr_mat=ghosted_csr_mat
        self.par_b = None
        self.xGhosted = None
        self.yGhosted = None
    def create(self, A):
        pass
    def mult(self, A, x, y):
        assert self.par_b is not None, "The parallel RHS vector par_b must be " \
                            "initialized before using the mult function"
        logEvent("Using SparseMatShell in LinearSolver matrix multiply")
        if self.xGhosted is None:
            self.xGhosted = self.par_b.duplicate()
            self.yGhosted = self.par_b.duplicate()
        self.xGhosted.setArray(x.getArray())
        self.xGhosted.ghostUpdateBegin(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        self.xGhosted.ghostUpdateEnd(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        self.yGhosted.zeroEntries()
        with self.xGhosted.localForm() as xlf, self.yGhosted.localForm() as ylf:
            self.ghosted_csr_mat.matvec(xlf.getArray(),ylf.getArray())
        y.setArray(self.yGhosted.getArray())

class OperatorShell(object):
    """ A base class for operator shells """
    def __init__(self):
        pass
    def create(self,A):
        pass
    def getSize(self):
        """
        Return the number of degrees of freedom for the operator.
        """
        raise NotImplementedError('You need to define a getSize ' \
                                  'method for your shell')

class ProductOperatorShell(OperatorShell):
    """ A base class for shell operators that apply multiplcation.

    Operators derived from this class should have working multiplication
    functions.
    """
    def __init__(self):
        pass
    def mult(self, A, x, y):
        raise NotImplementedError('You need to define a multiply' \
                                  'function for your shell')

class InvOperatorShell(OperatorShell):
    """ A base class for inverse operator shells

    Operators derived from this class should have working apply
    functions.
    """
    def __init__(self):
        pass

    @staticmethod
    def _create_tmp_vec(size):
        """ Creates an empty vector of given size.

        Arguments
        ---------
        size : int
            Size of the temporary vector.

        Returns
        -------
        vec : PETSc vector
        """
        tmp = p4pyPETSc.Vec().create()
        tmp.setType('mpi')
        tmp.setSizes(size)
        return tmp

    @staticmethod
    def _create_copy_vec(vec):
        """ Creates a copy of a petsc4py vector.

        Parameters
        ----------
        vec : :class:`petsc4py.Vec`

        Returns
        -------
        tmp : :class:`petsc4py.Vec`
        """
        tmp = p4pyPETSc.Vec().create()
        tmp.setType('mpi')
        tmp = vec.copy()
        return tmp

    def apply(self, A, x, y):
        raise NotImplementedError('You need to define an apply' \
                                  'method for your shell')

    def getSize(self):
        """ Returns the size of InvOperatorShell.
        
        Notes
        -----
        This acts a virtual method and must be implemented for 
        all inherited classes.
        """
        raise NotImplementedError()

    def create_petsc_ksp_obj(self,
                             petsc_option_prefix,
                             matrix_operator,
                             constant_null_space = False):
        """ Create a PETSc4py KSP object.

        Arguments
        ---------
        petsc_option_prefix : str
            PETSc commandline option prefix.
        matrix_operator : mat
            PETSc matrix object for the ksp class.
        null_space : bool
            True if the KSP object has a constant null space.

        Returns
        -------
        ksp_obj : PETSc ksp
        """
        ksp_obj = p4pyPETSc.KSP().create()
        ksp_obj.setOperators(matrix_operator,
                             matrix_operator)
        ksp_obj.setOptionsPrefix(petsc_option_prefix)

        if constant_null_space:
            const_nullspace_str = ''.join([petsc_option_prefix,
                                           'ksp_constant_null_space'])
            self.options.setValue(const_nullspace_str,'')
            matrix_operator.setNullSpace(self.const_null_space)
        ksp_obj.setFromOptions()

        ksp_obj.setUp()
        return ksp_obj

    def _create_constant_nullspace(self):
        """Initialize a constant null space. """
        self.const_null_space = p4pyPETSc.NullSpace().create(comm=p4pyPETSc.COMM_WORLD,
                                                             vectors = (),
                                                             constant = True)

    def _set_dirichlet_idx_set(self):
        """
        Initialize an index set of non-Dirichlet degrees of freedom.

        When the value of some degrees of freedom are known in
        advance it can be helfpul to remove these degrees of
        freedom from the inverse operator.  This function
        creates a PETSc4py index set of unknown degrees of freedom.
        """
        comm = Comm.get()
        # Assign number of unknowns
        num_dof = self.getSize()
        self.strong_dirichlet_DOF = [i for i in self.strong_dirichlet_DOF if i< num_dof]
        try:
            num_known_dof = len(self.strong_dirichlet_DOF)
        except AttributeError:
            print("ERROR - strong_dirichlet_DOF have not been " \
                  " assigned for this inverse operator object.")
            exit()

        num_unknown_dof = num_dof - num_known_dof
        # Use boolean mask to collect unknown DOF indices
        self.dof_indices = numpy.arange(num_dof,
                                        dtype = 'int32')
        known_dof_mask = numpy.ones(num_dof,
                                    dtype = bool)
        known_dof_mask[self.strong_dirichlet_DOF] = False

        self.unknown_dof_indices = self.dof_indices[known_dof_mask]
        self.known_dof_indices = self.dof_indices[~known_dof_mask]

        if comm.size() == 1:
            # Create PETSc4py index set of unknown DOF
            self.known_dof_is = p4pyPETSc.IS()
            self.known_dof_is.createGeneral(self.known_dof_indices,
                                            comm=p4pyPETSc.COMM_WORLD)
            self.unknown_dof_is = p4pyPETSc.IS()
            self.unknown_dof_is.createGeneral(self.unknown_dof_indices,
                                              comm=p4pyPETSc.COMM_WORLD)
        elif comm.size() > 1:
            self.global_known_dof_indices = [self.par_info.subdomain2global[i] for i in self.known_dof_indices]
            self.global_unknown_dof_indices = [self.par_info.subdomain2global[i] for i in self.unknown_dof_indices]

            self.known_dof_is = p4pyPETSc.IS()
            self.known_dof_is.createGeneral(self.global_known_dof_indices,
                                            comm=p4pyPETSc.COMM_WORLD)
            self.unknown_dof_is = p4pyPETSc.IS()
            self.unknown_dof_is.createGeneral(self.global_unknown_dof_indices,
                                              comm=p4pyPETSc.COMM_WORLD)

    def _converged_trueRes(self,ksp,its,rnorm):
        """ Function handle to feed to ksp's setConvergenceTest  """
        ksp.buildResidual(self.r_work)
        truenorm = self.r_work.norm()
        if its == 0:
            self.rnorm0 = truenorm
            # ARB - Leaving these log events in for future debugging purposes.
            # logEvent("NumericalAnalytics KSP_LSC_LaplaceResidual: %12.5e" %(truenorm) )
            # logEvent("NumericalAnalytics KSP_LSC_LaplaceResidual(relative): %12.5e" %(truenorm / self.rnorm0) )
            # logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
            #                                                                                   truenorm,
            #                                                                                   (truenorm/ self.rnorm0),
            #                                                                                   ksp.atol,
            #                                                                                   ksp.rtol))
            return False
        else:
            # ARB - Leaving these log events in for future debugging purposes.
            # logEvent("NumericalAnalytics KSP_LSC_LaplaceResidual: %12.5e" %(truenorm) )
            # logEvent("NumericalAnalytics KSP_LSC_LaplaceResidual(relative): %12.5e" %(truenorm / self.rnorm0) )
            # logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
            #                                                                                   truenorm,
            #                                                                                   (truenorm/ self.rnorm0),
            #                                                                                   ksp.atol,
            #                                                                                   ksp.rtol))
            if truenorm < self.rnorm0*ksp.rtol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_RTOL
            if truenorm < ksp.atol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_ATOL
        return False

class LSCInv_shell(InvOperatorShell):
    """ Shell class for the LSC Inverse Preconditioner

    This class creates a shell for the least-squares commutator (LSC)
    preconditioner, where
    :math:`M_{s}= (B \\hat{Q^{-1}_{v}} B^{'}) (B \\hat{Q^{-1}_{v}} F
    \\hat{Q^{-1}_{v}} B^{'})^{-1} (B \\hat{Q^{-1}_{v}} B^{'})`
    is used to approximate the Schur complement.
    """
    def __init__(self, Qv, B, Bt, F):
        """Initializes the LSC inverse operator.

        Parameters
        ----------
        Qv : petsc4py matrix object
            The diagonal elements of the velocity mass matrix.
        B : petsc4py matrix object
            The discrete divergence operator.
        Bt : petsc4py matrix object
            The discrete gradient operator.
        F : petsc4py matrix object
            The A-block of the linear system.
        """
        # TODO - Find a good way to assert that Qv is diagonal

        self.Qv = Qv
        self.B = B
        self.Bt = Bt
        self.F = F

        self._constructBQinvBt()
        self._options = p4pyPETSc.Options()

        if self._options.hasName('innerLSCsolver_BTinvBt_ksp_constant_null_space'):
            self._create_constant_nullspace()
            self.BQinvBt.setNullSpace(self.const_null_space)

        self.kspBQinvBt = p4pyPETSc.KSP().create()
        self.kspBQinvBt.setOperators(self.BQinvBt,self.BQinvBt)
        self.kspBQinvBt.setOptionsPrefix('innerLSCsolver_BTinvBt_')
        self.kspBQinvBt.pc.setUp()
        self.kspBQinvBt.setFromOptions()
        self.kspBQinvBt.setUp()

        # initialize solver for Qv
        self.kspQv = p4pyPETSc.KSP().create()
        self.kspQv.setOperators(self.Qv,self.Qv)
        self.kspQv.setOptionsPrefix('innerLSCsolver_T_')
        self.kspQv.setFromOptions()

        convergenceTest = 'r-true'
        if convergenceTest == 'r-true':
            self.r_work = self.BQinvBt.createVecLeft()
            self.rnorm0 = None
            self.kspBQinvBt.setConvergenceTest(self._converged_trueRes)
        else:
            self.r_work = None
        self.kspBQinvBt.setUp()

    def apply(self,A,x,y):
        """ Apply the LSC inverse operator

        Parameters
        ----------
        A : NULL
            A placeholder for internal function PETSc functions.
        x : :class:`p4pyPETSc.Vec`
            Vector which LSC operator is being applied to.

        Returns
        --------
        y : :class:`p4pyPETSc.Vec`
            Result of LSC acting on x.
        """
        # create temporary vectors
        B_sizes = self.B.getSizes()
        x_tmp = p4pyPETSc.Vec().create()
        x_tmp = x.copy()
        tmp1 = self._create_tmp_vec(B_sizes[0])
        tmp2 = self._create_tmp_vec(B_sizes[1])
        tmp3 = self._create_tmp_vec(B_sizes[1])

        if self._options.hasName('innerLSCsolver_BTinvBt_ksp_constant_null_space'):
            self.const_null_space.remove(x_tmp)
        self.kspBQinvBt.solve(x_tmp,tmp1)
        self.B.multTranspose(tmp1,tmp2)
        self.kspQv.solve(tmp2,tmp3)
        self.F.mult(tmp3,tmp2)
        self.kspQv.solve(tmp2,tmp3)
        self.B.mult(tmp3,tmp1)
        if self._options.hasName('innerLSCsolver_BTinvBt_ksp_constant_null_space'):
            self.const_null_space.remove(x_tmp)
        self.kspBQinvBt.solve(tmp1,y)
        assert numpy.isnan(y.norm())==False, "Applying the schur complement \
resulted in not-a-number."

    def _constructBQinvBt(self):
        """ Private method repsonsible for building BQinvBt """
        self.Qv_inv = petsc_create_diagonal_inv_matrix(self.Qv)
        QinvBt = self.Qv_inv.matMult(self.Bt)
        self.BQinvBt = self.B.matMult(QinvBt)

class MatrixShell(ProductOperatorShell):
    """ A shell class for a matrix. """
    def __init__(self,A):
        """
        Specifies a basic matrix shell.

        Parameters
        ----------
        A : matrix
            A petsc4py matrix object
        """
        self.A = A
    def mult(self,A,x,y):
        """
        Multiply the matrix and x.

        Parameters
        ----------
        A : matrix
            Dummy place holder for PETSc compatibility
        x : vector

        Returns
        -------
        y : vector
        """
        self.A.mult(x,y)

class MatrixInvShell(InvOperatorShell):
    """ A PETSc shell class for a inverse operator. """
    def __init__(self, A):
        """ Initializes operators and solvers for inverse operator.

        Parameters
        ----------
        A : PETSc matrix
            This is the matrix object used to construct the inverse.
        """
        self.A = A
        self.ksp = p4pyPETSc.KSP().create()
        self.ksp.setOperators(self.A,self.A)
        self.ksp.setType('preonly')
        self.ksp.pc.setType('lu')
        self.ksp.pc.setFactorSolverType('superlu_dist')
        self.ksp.setUp()

    def apply(self,A,x,y):
        """ Apply the inverse pressure mass matrix.

        Parameters
        ----------
        A : matrix
            Dummy place holder for PETSc compatibility
        x : vector

        Returns
        -------
        y : vector
        """
        self.ksp.solve(x,y)

class SpInv_shell(InvOperatorShell):
    r""" Shell class for the SIMPLE preconditioner which applies the
    following action:

    .. math::
        \hat{S}^{-1} = (A_{11} - A_{01} \text{diag}(A_{00}) A_{10})^{-1}

    where :math:`A_{ij}` are sub-blocks of the global saddle point system.

    Parameters
    ----------
    A00: :class:`p4pyPETSc.Mat`
        The A00 block of the global saddle point system.
    A01: :class:`p4pyPETSc.Mat`
        The A01 block of the global saddle point system.
    A10: :class:`p4pyPETSc.Mat`
        The A10 block of the global saddle point system.
    A11: :class:`p4pyPETSc.Mat`
        The A11 block of the global saddle point system.
    use_constant_null_space: bool
        Indicates whether a constant null space should be used.  See
        note below.

    Notes
    -----
    For Stokes or Navier-Stokes systems, the :math:`S` operator
    resembles a Laplcian matrix on the pressure.  In cases where the
    global saddle point system uses pure Dirichlet boundary
    conditions, the :math:`S^{-1}` operator has a constant null
    space.  Since most saddle-point simulations of interest do not
    have pure Dirichlet conditions, the `constNullSpace` flag defaults
    to false.  Having the null space set to false when the global
    problem uses pure Dirichlet boundary conditions will likely result
    in poor solver performance or failure.
    """
    def __init__(self, A00, A11, A01, A10, constNullSpace):
        self.A00 = A00
        self.A11 = A11
        self.A01 = A01
        self.A10 = A10
        self.constNullSpace = constNullSpace
        self._create_Sp()
        self._options = p4pyPETSc.Options()

        self.kspSp = p4pyPETSc.KSP().create()
        self.kspSp.setOperators(self.Sp,self.Sp)
        self.kspSp.setOptionsPrefix('innerSpsolver_')
        self.kspSp.setFromOptions()
        if self.constNullSpace:
            self._create_constant_nullspace()
            self.Sp.setNullSpace(self.const_null_space)
        self.kspSp.setUp()

    def apply(self,A,x,y):
        """ Applies the :math:`S_{p}` operator

        Parameters
        ----------
        A : None
            Dummy argument for PETSc interface
        x : :class:`p4pyPETSc.Vec`
            Vector to which :math:`S` is applied

        Returns
        -------
        y : :class:`p4pyPETSc.Vec`
            Result of :math:`S^{-1}x`
        """
        tmp1 = p4pyPETSc.Vec().create()
        tmp1 = x.copy()
        if self.constNullSpace:
            self.const_null_space.remove(tmp1)
        self.kspSp.solve(tmp1,y)
        assert numpy.isnan(y.norm())==False, "Applying the schur complement \
        resulted in not-a-number."

    def _create_Sp(self):
        self.A00_inv = petsc_create_diagonal_inv_matrix(self.A00)
        A00_invBt = self.A00_inv.matMult(self.A01)
        self.Sp = self.A10.matMult(A00_invBt)
        self.Sp.aypx(-1.,self.A11)

class TwoPhase_PCDInv_shell(InvOperatorShell):
    r""" Shell class for the two-phase PCD preconditioner.  The
    two-phase PCD_inverse shell applies the following operator.

    .. math::

        \hat{S}^{-1} = (Q^{(1 / \mu)})^{-1} + (A_{p}^{(1 / \rho)})^{-1}
        (N_{p}^{(\rho)} + \dfrac{\alpha}{\Delta t} Q^{(\rho)} )
        (Q^{(\rho)})^{-1}

    where :math:`Q^{(1 / \mu)}` and :math:`Q^{(\rho)}` denote the pressure
    mass matrix scaled by the inverse dynamic viscosity and density
    respectively, :math:`(A_{p}^{(1 / \rho)})^{-1}`
    denotes the pressure Laplacian scaled by inverse density, and
    :math:`N_{p}^{(\rho)}` denotes the pressure advection operator scaled by
    the density, and :math:`\alpha` is a binary operator indicating
    whether the problem is temporal or steady state.
    """
    def __init__(self,
                 Qp_visc,
                 Qp_dens,
                 Ap_rho,
                 Np_rho,
                 alpha = False,
                 delta_t = 0,
                 num_chebyshev_its = 0,
                 strong_dirichlet_DOF = [],
                 laplace_null_space = False,
                 par_info=None):
        """ Initialize the two-phase PCD inverse operator.

        Parameters
        ----------
        Qp_visc : petsc4py matrix
            The pressure mass matrix with dynamic viscocity
            scaling.
        Qp_dens : petsc4py matrix
            The pressure mass matrix with density scaling.
        Ap_rho : petsc4py matrix
            The pressure Laplacian scaled with density scaling.
        Np_rho : petsc4py matrix
            The pressure advection operator with inverse density
            scaling.
        alpha : binary
            True if problem is temporal, False if problem is steady
            state.
        delta_t : float
            Time step parameter.
        num_chebyshev_its : int
            Number of chebyshev iteration steps to take. (0 indicates
            the chebyshev semi iteration is not used)
        strong_dirichlet_DOF : lst
            List of DOF with known, strongly enforced values.
        laplace_null_space : binary
            Indicates whether the pressure Laplace matrix has a
            null space or not.
        par_info : ParInfoClass
            Provides parallel info.
        """
        from . import LinearSolvers as LS

        # Set attributes
        self.Qp_visc = Qp_visc
        self.Qp_dens = Qp_dens
        self.Ap_rho = Ap_rho
        self.Np_rho = Np_rho
        self.alpha = alpha
        self.delta_t = delta_t
        self.num_chebyshev_its = num_chebyshev_its
        self.strong_dirichlet_DOF = strong_dirichlet_DOF
        self.laplace_null_space = laplace_null_space
        self.par_info = par_info

        self.options = p4pyPETSc.Options()
        self._create_constant_nullspace()
        self._set_dirichlet_idx_set()

        self.kspAp_rho = self.create_petsc_ksp_obj('innerTPPCDsolver_Ap_rho_',
                                                   self.Ap_rho,
                                                   self.laplace_null_space)

        self.kspAp_rho.getOperators()[0].zeroRows(self.known_dof_is)

        if self.num_chebyshev_its:
            self.Qp_visc = LS.ChebyshevSemiIteration(self.Qp_visc,
                                                     0.5,
                                                     2.0)
            self.Qp_dens = LS.ChebyshevSemiIteration(self.Qp_dens,
                                                     0.5,
                                                     2.0)
        else:
            pass
            # Using ksp objects for the lumped mass matrices is much
            # slower than pointwise division.
            # self.kspQp_visc = self.create_petsc_ksp_obj('innerTPPCDsolver_Qp_visc_',
            #                                             self.Qp_visc)
            # self.kspQp_dens = self.create_petsc_ksp_obj('innerTPPCDsolver_Qp_dens_',
            #                                             self.Qp_dens)

    def getSize(self):
        """ Return the total number of DOF for the shell problem. """
        return self.Ap_rho.getSizes()[0][0]

    def apply(self,A,x,y):
        """
        Applies the two-phase pressure-convection-diffusion
        Schur complement approximation.

        Parameters
        ----------
        A : None
            Dummy variabled needed to interface with PETSc
        x : petsc4py vector
            Vector to which operator is applied

        Returns
        -------
        y : petsc4py vector
            Result of operator acting on x.

        Notes
        -----
        When strong Dirichlet conditions are enforced on the pressure,
        the PCD operator is applied to the set of unknowns that do not
        have Dirichlet boundary conditions.  At the end, the solution
        is then loaded into the original y-vector.
        """
        comm = Comm.get()
        x_tmp = self._create_copy_vec(x)
        tmp1 = self._create_copy_vec(x_tmp)
        tmp2 = self._create_copy_vec(x_tmp)

        if self.num_chebyshev_its:
            self.Qp_visc.apply(x_tmp,
                               y,
                               self.num_chebyshev_its)
            self.Qp_dens.apply(x_tmp,
                               tmp1,
                               self.num_chebyshev_its)
        else:
            y.pointwiseDivide(x_tmp,self.Qp_visc.getDiagonal())
            tmp1.pointwiseDivide(x_tmp,self.Qp_dens.getDiagonal())
            # Pointwise divide appears to be much faster than ksp.
            # self.kspQp_visc.solve(x_tmp,y)
            # self.kspQp_dens.solve(x_tmp,tmp1)

        self.Np_rho.mult(tmp1,tmp2)

        if self.alpha is True:
            tmp2.axpy(1./self.delta_t,x_tmp)

        if self.options.hasName('innerTPPCDsolver_Ap_rho_ksp_constant_null_space'):
            self.const_null_space.remove(tmp2)

        zero_array = numpy.zeros(len(self.known_dof_is.getIndices()))

        tmp2.setValues(self.known_dof_is.getIndices(),zero_array)
        tmp2.assemblyBegin()
        tmp2.assemblyEnd()

        self.kspAp_rho.solve(tmp2, tmp1)
        y.axpy(1.,tmp1)
        y.setValues(self.known_dof_is.getIndices(),zero_array)
        y.assemblyBegin()
        y.assemblyEnd()

        assert numpy.isnan(y.norm())==False, "Applying the schur complement \
        resulted in not-a-number."

def l2Norm(x):
    """
    Compute the parallel :math:`l_2` norm
    """
    return math.sqrt(globalSum(numpy.dot(x,x)))


def l1Norm(x):
    """
    Compute the parallel :math:`l_1` norm

    The :math:`l_1` norm of a vector :math:`\\mathbf{x} \\in
    \\mathbb{R}^n` is

    .. math::

       \\| \\mathbf{x} \\|_{1} = \\sum_{i=0} |x_i|

    If Python is running in parallel, then the sum is over all
    dimensions on all processors so that the input must not contain
    "ghost" entries.

    This implemtation works for a distributed array with no ghost
    components (each component must be on a single processor).

    :param x: numpy array of length n
    :return: float
    """
    return globalSum(numpy.sum(numpy.abs(x)))


def lInfNorm(x):
    """
    Compute the parallel :math:`l_{\\infty}` norm

    The :math:`l_{\\infty}` norm of a vector :math:`\\mathbf{x} \\in
    \\mathbb{R}^n` is

    .. math::

       \\|x\\|_{\\infty} = \\max_i |x_i|

    This implemtation works for a distributed array with no ghost
    components (each component must be on a single processor).

    :param x: numpy array of length n
    :return: float
    """
    return globalMax(numpy.linalg.norm(x,numpy.inf))


def wDot(x,y,h):
    """
    Compute the parallel weighted dot product of vectors x and y using
    weight vector h.

    The weighted dot product is defined for a weight vector
    :math:`\\mathbf{h}` as

    .. math::

       (\\mathbf{x},\\mathbf{y})_h = \\sum_{i} h_{i} x_{i} y_{i}

    All weight vector components should be positive.

    :param x,y,h: numpy arrays for vectors and weight
    :return: the weighted dot product
    """
    return globalSum(numpy.sum(x*y*h))

def wl2Norm(x,h):
    """
    Compute the parallel weighted l_2 norm with weight h
    """
    return math.sqrt(wDot(x,x,h))  # wDot is already the global (reduced) sum


def wl1Norm(x,h):
    """
    Compute the parallel weighted l_1 norm with weight h
    """
    return globalSum(numpy.sum(numpy.abs(h*x)))


def wlInfNorm(x,h):
    """
    Compute the parallel weighted l_{\\infty} norm with weight h
    """
    return globalMax(numpy.linalg.norm(h*x,numpy.inf))

def energyDot(x,y,A):
    """
    Compute the "energy" dot product x^t A y (not parallel)
    """
    return numpy.dot(numpy.dot(x,A),y)

def energyNorm(x,A):
    """
    Compute the "energy" norm x^t A x (not parallel)
    """
    return math.sqrt(energyDot(x,x,A))

def l2NormAvg(x):
    """
    Compute the arithmetic averaged l_2 norm (root mean squared norm)
    """
    scale = 1.0/globalSum(len(x.flat))
    return math.sqrt(scale*globalSum(numpy.dot(x,x)))


rmsNorm = l2NormAvg


def l2Norm_local(x):
    """
    Compute the l_2 norm for just local (processor) system  (not parallel)
    """
    return math.sqrt(numpy.dot(x,x))


class WeightedNorm(object):
    """
    Compute the weighted norm for time step control (not currently parallel)
    """
    def __init__(self,shape,atol,rtol):
        self.shape = shape
        self.dim = sum(self.shape)
        self.atol= atol
        self.rtol= rtol
        self.weight = numpy.ones(shape,'d')
        self.tmp    = numpy.ones(shape,'d')
    def setWeight(self,y):
        self.weight[:] = numpy.absolute(y)
        self.weight   *= self.rtol
        self.weight   += self.atol
    def norm(self,y,type):
        self.tmp[:] = y
        self.tmp /= self.weight
        value = numpy.linalg.norm(self.tmp.flat,type)
        return value/self.dim


if __name__ == '__main__':
    import doctest
    doctest.testmod()


# def test_MGV():
#     n=2**8 + 1
#     h =1.0/(n-1.0)
#     freq=10
#     u = numpy.random.uniform(0,1,(n))
#     u[0]=0.0
#     u[n-1]=0.0
#     x = numpy.arange(0,1.0+h,h)
#     AList=[]
#     N=n
#     pList=[]
#     rList=[]
#     resList=[]
#     while N >= 3:
#         resList.append(Vec(N-2))
#         A = dict()#SparseMat(N-2,N-2,3*(N-2),sym=True)
#         H = 1.0/(N-1.0)
#         #beginAssembly(A)
#         for i in range(N-2):
#             A[(i,i)] = 2.0/H**2
#             if i > 0:
#                 A[(i,i-1)] = -1.0/H**2
#             if i < N-3:
#                 A[(i,i+1)] = -1.0/H**2
#         #endAssembly(A)
#         AList.append(SparseMatFromDict(N-2,N-2,A)[0])
#         cN = (N - 1)/2 + 1
#         r = dict()#SparseMat(cN-2,N-2,3*(N-2))
#         p = dict()#SparseMat(N-2,cN-2,3*(N-2))
#         for i in range(cN-2):
#             r[(i,2*i)]   = 1.0/4.0
#             r[(i,2*i+1)] = 2.0/4.0
#             r[(i,2*i+2)] = 1.0/4.0
#             p[(2*i,i)] = 1.0/2.0
#             p[(2*i+1,i)]= 2.0/2.0
#             p[(2*i+2,i)]= 1.0/2.0
#         #r.to_csr()
#         print cN-2,N-2,r.keys()
#         if cN-2 > 0:
#             rList.append(SparseMatFromDict(cN-2,N-2,r)[0])
#         else:
#             rList.append(None)
#         #p.to_csr()
#         pList.append(SparseMatFromDict(N-2,cN-2,p)[0])
#         N = cN
#     class Jacobi:
#         def __init__(self,A):
#             self.A=A
#             self.n=A.shape[0]
#             self.M=Vec(self.n)
#             for i in range(self.n):
#                 self.M[i]=1.0/A[i,i]
#             self.res=Vec(self.n)
#             self.dx=Vec(self.n)
#         def apply(self,w,jits,b,x):
#             self.A.matvec(x,self.res)
#             self.res-=b
#             for it in range(jits):
#                 self.dx[:] = self.M*self.res
#                 self.dx*=w
#                 x -= self.dx
#                 self.A.matvec(x,self.res)
#                 self.res -= b
#     jacobiList=[]
#     for A in AList:
#         jacobiList.append(Jacobi(A))
#     jits = 3
#     w = 2.0/3.0
#     class MGV:
#         def __init__(self,smootherList,AList,pList,rList,resList):
#             self.AList = AList
#             self.pList = pList
#             self.rList = rList
#             self.resList = resList
#             self.xList=[]
#             self.vList=[]
#             self.bList=[]
#             self.gpList=[]
#             for res in resList:
#                 self.xList.append(Vec(len(res)))
#                 self.vList.append(Vec(len(res)))
#                 self.bList.append(Vec(len(res)))
#             self.smootherList = smootherList

#         def apply(self,w,nsPre,nsPost,level,b,x):
#             logEvent("Level = "+`level`)
#             if level == len(self.AList)-1:
#                 self.smootherList[level].apply(1.0,1,b,x)
#             else:
#                 #smooth
#                 self.smootherList[level].apply(w,nsPre,b,x)
#                 #restrict the defect
#                 self.rList[level].matvec(self.smootherList[level].res,self.bList[level+1])
#                 #V-cycle on the error equation
#                 self.xList[level+1][:]=0.0
#                 self.apply(w,nsPre,nsPost,level+1,self.bList[level+1],self.xList[level+1])
#                 #prolong
#                 self.pList[level].matvec(self.xList[level+1],self.vList[level])
#                 #correct
#                 x-=self.vList[level]
#                 #smooth
#                 self.smootherList[level].apply(w,nsPost,b,x)
#                 self.resList[level][:]=self.smootherList[level].res
#     mgv = MGV(jacobiList,AList,pList,rList,resList)
#     rnorm=1.0
#     mgits = 0
#     while rnorm > 1.0e-10 and mgits < 20:
#         mgits +=1
#         mgv.apply(w,jits,jits,0,f[1:n-1],u[1:n-1])
#         rnorm = l2Norm(resList[0])


def nodalProlongationMat(nodeParents,coarseNodeOffsets,fineNodeOffsets,comm=None,
                         inheritedOnly=False,blockSize=1):
    """The coarse-to-fine prolongation for a C0 nodal space, as a PETSc Mat.

    Uniform refinement keeps each coarse node's number and adds the rest at
    edge midpoints, so every fine node is described by a pair of coarse nodes
    (see MeshTools.MultilevelMesh.calculateNodeParents). The prolongation then
    needs no geometry at all: a fine node the refinement inherited copies its
    coarse value, and one it added averages the two coarse nodes whose edge it
    bisects.

    That is what makes this work in parallel. Row i depends only on fine node
    i's two parents, so a rank can fill every row it owns with no communication
    and no access to the coarse mesh -- in contrast to building the operator
    geometrically, which needs coarse basis functions evaluated at fine
    interpolation points and so needs the coarse elements to be reachable.

    Parameters
    ----------
    nodeParents : numpy.ndarray
        (nNodes_fine, 2), in global coarse node numbers, as produced by
        calculateNodeParents.
    coarseNodeOffsets, fineNodeOffsets : numpy.ndarray
        nodeOffsets_subdomain_owned for the two levels: rank r owns global
        nodes [offsets[r], offsets[r+1]). Node-based partitioning numbers the
        owned nodes contiguously in rank order, which is also how PETSc
        distributes rows, so no index translation is needed.

    inheritedOnly : bool
        Keep only the rows of the nodes the refinement inherited, leaving the
        rows of added nodes empty. That is the injection of the coarse
        nodes into the fine ones, whose transpose picks the fine values
        sitting on coarse nodes -- the interpolation the multilevel
        solvers use to restrict a *solution* (proteus's interp_bcList).

    blockSize : int
        Number of components interleaved in the vectors the Mat acts on, so
        that component c of node i is row ``blockSize*i + c``. The result is
        then P (x) I -- every component transferred by the same nodal
        operator -- laid out the way the stacked system's vectors and matrices
        are, which is what PETSc's multigrid needs for a multi-component
        problem. Nodal spaces are always interleaved: the components share a
        dof count, and the system is interleaved whenever weak Dirichlet
        conditions apply, which parallel runs require.

    Returns
    -------
    petsc4py Mat with fine rows and coarse columns.
    """
    bs = int(blockSize)
    assert bs >= 1, bs
    if comm is None:
        comm = Comm.get()
    rank = comm.rank()
    fine_lo,fine_hi = int(fineNodeOffsets[rank]),int(fineNodeOffsets[rank+1])
    coarse_lo,coarse_hi = int(coarseNodeOffsets[rank]),int(coarseNodeOffsets[rank+1])
    nFine_global = int(fineNodeOffsets[-1])
    nCoarse_global = int(coarseNodeOffsets[-1])
    assert nodeParents.shape[0] == nFine_global, (
        "nodeParents has %i rows but the fine level has %i nodes"
        % (nodeParents.shape[0],nFine_global))

    owned = nodeParents[fine_lo:fine_hi]
    inherited = owned[:,0] == owned[:,1]

    #at most two coarse nodes per fine node; splitting the count into the part
    #this rank owns and the rest keeps PETSc from over-allocating
    local = numpy.where((owned >= coarse_lo) & (owned < coarse_hi),1,0)
    d_nnz = numpy.where(inherited,local[:,0],local.sum(axis=1)).astype('i')
    o_nnz = numpy.where(inherited,1-local[:,0],2-local.sum(axis=1)).astype('i')
    if inheritedOnly:
        d_nnz = numpy.where(inherited,d_nnz,0).astype('i')
        o_nnz = numpy.where(inherited,o_nnz,0).astype('i')
    if bs > 1:
        #each node's row is repeated once per component, with the same pattern
        d_nnz = numpy.repeat(d_nnz,bs).astype('i')
        o_nnz = numpy.repeat(o_nnz,bs).astype('i')
    prolong = newTransferMat(comm,[[bs*(fine_hi-fine_lo),bs*nFine_global],
                                   [bs*(coarse_hi-coarse_lo),bs*nCoarse_global]],
                             d_nnz,o_nnz)

    for i in range(fine_hi-fine_lo):
        row = fine_lo + i
        a,b = int(owned[i,0]),int(owned[i,1])
        if a == b:
            for c in range(bs):
                prolong.setValues([bs*row+c],[bs*a+c],[1.0])
        elif not inheritedOnly:
            for c in range(bs):
                prolong.setValues([bs*row+c],[bs*a+c,bs*b+c],[0.5,0.5])
    prolong.assemblyBegin()
    prolong.assemblyEnd()
    return prolong


def vertexProlongationMat(rowptr,colind,nzval,coarseNodeOffsets,fineNodeOffsets,
                          comm=None,inheritedOnly=False,blockSize=1):
    """A vertex-to-vertex prolongation, as a distributed PETSc Mat.

    The general form of nodalProlongationMat: the same Mat, but each fine node
    may have any number of coarse parents with any weights, which is what a
    quadrilateral's cell centre (four parents) or a hexahedron's body centre
    (eight) needs. It takes the prolongation as a CSR matrix in the
    *partitioned* numbering -- see MultilevelMesh.partitionedVertexInterpolation
    -- and, like nodalProlongationMat, has each rank fill only the rows of the
    fine nodes it owns, which needs no communication.

    Parameters
    ----------
    rowptr, colind, nzval : numpy.ndarray
        The global prolongation, one row per fine node, in global partitioned
        numbers -- or only the rows of the fine nodes this rank owns, in order,
        as a hierarchy generated in pieces gives them.
    coarseNodeOffsets, fineNodeOffsets : numpy.ndarray
        nodeOffsets_subdomain_owned for the two levels.
    inheritedOnly : bool
        Keep only the rows that copy a single coarse value with weight one --
        the nodes a refinement inherits. That is the injection whose transpose
        picks the fine values sitting on coarse nodes.
    blockSize : int
        Interleaved components, as in nodalProlongationMat: component c of node
        i is entry ``blockSize*i + c``.
    """
    bs = int(blockSize)
    if comm is None:
        comm = Comm.get()
    rank = comm.rank()
    fine_lo,fine_hi = int(fineNodeOffsets[rank]),int(fineNodeOffsets[rank+1])
    coarse_lo,coarse_hi = int(coarseNodeOffsets[rank]),int(coarseNodeOffsets[rank+1])
    nFine_global = int(fineNodeOffsets[-1])
    nCoarse_global = int(coarseNodeOffsets[-1])
    if rowptr.shape[0] == nFine_global+1:
        local = lambda r: r
    else:
        assert rowptr.shape[0] == fine_hi-fine_lo+1, (
            "the prolongation has %i rows but the fine level has %i nodes, %i of them owned here"
            % (rowptr.shape[0]-1,nFine_global,fine_hi-fine_lo))
        local = lambda r: r-fine_lo
    rows = range(fine_lo,fine_hi)
    keep = numpy.ones((fine_hi-fine_lo,),dtype=bool)
    if inheritedOnly:
        for i,r in enumerate(rows):
            lo,hi = rowptr[local(r)],rowptr[local(r)+1]
            keep[i] = (hi-lo == 1) and abs(nzval[lo]-1.0) <= 1.0e-12
    d_nnz = numpy.zeros((fine_hi-fine_lo,),'i')
    o_nnz = numpy.zeros((fine_hi-fine_lo,),'i')
    for i,r in enumerate(rows):
        if keep[i]:
            cols = colind[rowptr[local(r)]:rowptr[local(r)+1]]
            inside = int(((cols >= coarse_lo) & (cols < coarse_hi)).sum())
            d_nnz[i] = inside
            o_nnz[i] = cols.shape[0]-inside

    if bs > 1:
        d_nnz = numpy.repeat(d_nnz,bs).astype('i')
        o_nnz = numpy.repeat(o_nnz,bs).astype('i')
    prolong = newTransferMat(comm,[[bs*(fine_hi-fine_lo),bs*nFine_global],
                                   [bs*(coarse_hi-coarse_lo),bs*nCoarse_global]],
                             d_nnz,o_nnz)
    for i,r in enumerate(rows):
        if not keep[i]:
            continue
        cols = colind[rowptr[local(r)]:rowptr[local(r)+1]]
        vals = nzval[rowptr[local(r)]:rowptr[local(r)+1]]
        for c in range(bs):
            prolong.setValues([bs*r+c],(bs*cols+c).tolist(),vals.tolist())
    prolong.assemblyBegin()
    prolong.assemblyEnd()
    return prolong


def filterCSR(rowptr,colind,nzval,select,transform=None):
    """The same CSR matrix keeping only the entries ``select(nzval)`` accepts."""
    rowptr = numpy.asarray(rowptr); colind = numpy.asarray(colind); nzval = numpy.asarray(nzval)
    mask = select(nzval)
    rows = numpy.repeat(numpy.arange(rowptr.size-1),numpy.diff(rowptr))
    counts = numpy.bincount(rows[mask],minlength=rowptr.size-1)
    newptr = numpy.concatenate([[0],numpy.cumsum(counts)]).astype(rowptr.dtype)
    vals = nzval[mask]
    return (newptr,colind[mask],vals if transform is None else transform(vals))


def subdomainProlongationMat(rowptr,colind,nzval,fineSubdomain2global,coarseSubdomain2global,
                             fineDofOffsets,coarseDofOffsets,comm=None,blockSize=1,allowEmptyRows=False):
    """A prolongation given as each rank's own rows, as a distributed PETSc Mat.

    The matrix a rank holds is between its *subdomain* dofs: fine rows and
    coarse columns numbered locally, owned dofs first. Only the owned rows are
    complete -- a ghost row is assembled from elements the rank holds
    partially -- so those are what is kept, translated to global dof numbers
    through the two subdomain2global maps. Nothing is gathered or replicated:
    every column a row needs was already in the coarse subdomain, which is what
    inheriting the coarse partition guarantees.

    fineDofOffsets/coarseDofOffsets are the dof maps' dof_offsets_subdomain_owned.
    blockSize interleaves components as in nodalProlongationMat. An empty owned
    row is an error for a prolongation, where it means a coarse parent is
    missing, but expected for an injection, which only has rows for the dofs a
    refinement inherited: pass allowEmptyRows=True for those.
    """
    bs = int(blockSize)
    if comm is None:
        comm = Comm.get()
    rank = comm.rank()
    fine_lo,fine_hi = int(fineDofOffsets[rank]),int(fineDofOffsets[rank+1])
    coarse_lo,coarse_hi = int(coarseDofOffsets[rank]),int(coarseDofOffsets[rank+1])
    nOwned = fine_hi-fine_lo
    fineGlobal = numpy.asarray(fineSubdomain2global)[:nOwned]
    assert nOwned == 0 or (fineGlobal.min() == fine_lo and fineGlobal.max() == fine_hi-1
                           and numpy.unique(fineGlobal).size == nOwned), (
        "the owned dofs are not the first ones of the subdomain, or not contiguous")
    coarseGlobal = numpy.asarray(coarseSubdomain2global)
    d_nnz = numpy.zeros((nOwned,),'i')
    o_nnz = numpy.zeros((nOwned,),'i')
    rows = []
    empty = []
    for i in range(nOwned):
        lo,hi = int(rowptr[i]),int(rowptr[i+1])
        cols = coarseGlobal[colind[lo:hi]]
        vals = numpy.asarray(nzval[lo:hi])
        if hi == lo:
            empty.append(int(fineGlobal[i]))
        inside = int(((cols >= coarse_lo) & (cols < coarse_hi)).sum())
        d_nnz[i] = inside
        o_nnz[i] = cols.shape[0]-inside
        rows.append((int(fineGlobal[i]),cols,vals))
    #Assembling the Mat is collective, so a rank that finds its rows incomplete
    #cannot just raise: the others would wait for it forever. Agree first.
    if not allowEmptyRows:
        mpi = comm.comm.tompi4py()
        problems = mpi.allgather(empty)
        if any(problems):
            raise ValueError(
                "ranks %s own dofs with an empty prolongation row (for example %s): the "
                "coarse parent is not on the rank, so the partitions are not "
                "level-consistent -- see MultilevelMesh.partitionLevels"
                % ([r for r,p in enumerate(problems) if p],
                   [p[0] for p in problems if p][:3]))
    if bs > 1:
        d_nnz = numpy.repeat(d_nnz,bs).astype('i')
        o_nnz = numpy.repeat(o_nnz,bs).astype('i')
    prolong = newTransferMat(comm,[[bs*nOwned,bs*int(fineDofOffsets[-1])],
                                   [bs*(coarse_hi-coarse_lo),bs*int(coarseDofOffsets[-1])]],
                             d_nnz,o_nnz)
    for (r,cols,vals) in rows:
        for c in range(bs):
            prolong.setValues([bs*r+c],(bs*cols+c).tolist(),vals.tolist())
    prolong.assemblyBegin()
    prolong.assemblyEnd()
    return prolong


def nodalRestrictionScaling(prolong):
    """Row sums of the restriction, for restricting *solutions* rather than residuals.

    Restriction of a residual is the adjoint of prolongation, P^T. Restricting
    a solution needs the row-sum-normalized operator the multilevel literature
    writes as I-tilde (Kees et al. 2006, section 2.3), because P^T sums the
    contributions of every fine node that sees a coarse node instead of
    averaging them.

    The row sums are P^T applied to a vector of ones, which PETSc assembles
    across ranks itself -- so the global accumulation a hand-rolled version
    would need on partition boundaries comes for free, and cannot be forgotten.
    """
    ones = prolong.createVecLeft()
    ones.set(1.0)
    rowSums = prolong.createVecRight()
    prolong.multTranspose(ones,rowSums)
    return rowSums


class ParTransferOperator(object):
    """A distributed transfer operator that behaves like a proteus SparseMat.

    The multilevel solvers call ``prolong.matvec(x,y)`` on plain numpy arrays
    holding a subdomain's unknowns, owned entries first and ghosts after. A
    PETSc Mat works on the owned range instead, so this copies between the two
    and refreshes the result's ghost entries, which the solvers read straight
    away when they form the next residual.

    ``shape`` reports the subdomain sizes rather than the global ones, because
    that is what the callers size their own work vectors from.
    """
    def __init__(self,mat,columnTemplate,rowTemplate):
        """mat maps the coarse level to the fine one.

        columnTemplate and rowTemplate are ParVecs carrying the coarse and fine
        parallel layouts; their arrays are not used, only their shapes.
        """
        self.mat = mat
        self.columnTemplate = columnTemplate
        self.rowTemplate = rowTemplate
        self.nOwnedColumns = columnTemplate.dim_proc
        self.nOwnedRows = rowTemplate.dim_proc
        self.shape = [rowTemplate.proteus_array.size,
                      columnTemplate.proteus_array.size]
        self._columnVec = mat.createVecRight()
        self._rowVec = mat.createVecLeft()
        self._ghostRefresh = {}

    def _parVecFor(self,array,template):
        """A ParVec over this caller's array, so its ghosts can be refreshed.

        The solvers reuse a handful of fixed buffers, so keep one per array
        rather than rebuilding it on every matvec.
        """
        if getattr(template,'nghosts',None) is None:
            return None
        cached = self._ghostRefresh.get(id(array))
        if cached is None or cached.proteus_array is not array:
            if array.size != template.proteus_array.size:
                return None
            cached = self._ghostRefresh[id(array)] = template.duplicate_with_array(array)
        return cached

    def matvec(self,x,y):
        assert x.size == self.shape[1], (x.size,self.shape)
        assert y.size == self.shape[0], (y.size,self.shape)
        self._columnVec.getArray()[:] = x[:self.nOwnedColumns]
        self.mat.mult(self._columnVec,self._rowVec)
        y[:self.nOwnedRows] = self._rowVec.getArray()
        refresh = self._parVecFor(y,self.rowTemplate)
        if refresh is not None:
            refresh.scatter_forward_insert()
        return y


class ParTransposeTransferOperator(ParTransferOperator):
    """The adjoint of a ParTransferOperator, for restricting residuals.

    Restriction is the transpose of prolongation, so this shares the same Mat
    rather than assembling a second one; the roles of the two layouts swap.
    """
    def __init__(self,mat,columnTemplate,rowTemplate):
        ParTransferOperator.__init__(self,mat,columnTemplate,rowTemplate)
        self.shape = [columnTemplate.proteus_array.size,
                      rowTemplate.proteus_array.size]
        self.nOwnedColumns,self.nOwnedRows = self.nOwnedRows,self.nOwnedColumns
        self.columnTemplate,self.rowTemplate = rowTemplate,columnTemplate

    def matvec(self,x,y):
        assert x.size == self.shape[1], (x.size,self.shape)
        assert y.size == self.shape[0], (y.size,self.shape)
        self._rowVec.getArray()[:] = x[:self.nOwnedColumns]
        self.mat.multTranspose(self._rowVec,self._columnVec)
        y[:self.nOwnedRows] = self._columnVec.getArray()
        refresh = self._parVecFor(y,self.rowTemplate)
        if refresh is not None:
            refresh.scatter_forward_insert()
        return y


class ParScaledTransposeTransferOperator(ParTransposeTransferOperator):
    """The row-sum-normalized restriction, for restricting *solutions*.

    P^T adds up the contributions of every fine node that sees a coarse node,
    which is right for residuals and wrong for a solution, so the multilevel
    literature (Kees et al. 2006, section 2.3) scales each row by its sum. The
    scaling is the vector nodalRestrictionScaling returns, one entry per owned
    coarse node; applying it after the transpose product keeps the whole thing
    a single distributed operation, so ghosts are refreshed once, last.
    """
    def __init__(self,mat,columnTemplate,rowTemplate,rowSums):
        ParTransposeTransferOperator.__init__(self,mat,columnTemplate,rowTemplate)
        self.rowSums = numpy.array(rowSums.getArray(),copy=True)
        assert self.rowSums.size == self.nOwnedRows, (self.rowSums.size,self.nOwnedRows)
        assert (self.rowSums > 0.0).all(), "a coarse node has no fine node under it"

    def matvec(self,x,y):
        assert x.size == self.shape[1], (x.size,self.shape)
        assert y.size == self.shape[0], (y.size,self.shape)
        self._rowVec.getArray()[:] = x[:self.nOwnedColumns]
        self.mat.multTranspose(self._rowVec,self._columnVec)
        y[:self.nOwnedRows] = self._columnVec.getArray()/self.rowSums
        refresh = self._parVecFor(y,self.rowTemplate)
        if refresh is not None:
            refresh.scatter_forward_insert()
        return y


if not HAVE_PETSC4PY:
    #the classes the rest of proteus constructs by these names
    ParVec_petsc4py = ParVec_mpi4py
    ParMat_petsc4py = ParMat_mpi4py


def strongDirichletNonAuthorityDofs(model, par_r):
    """The local dofs whose strong-Dirichlet row this rank must not contribute.

    With ``forceStrongConditions`` every rank replaces the rows of the dofs in its own
    ``dirichletConditionsForceDOF`` by ``u - g`` (residual) and an identity row
    (Jacobian), and the reverse add then sums the copies of a shared dof. Whether a
    dof is in a rank's dictionary depends on the boundary faces of that rank's own
    elements, so a node on the rank cut can be Dirichlet on one rank and free on
    another: the assembled row was an identity plus a partial PDE row, and where
    both ranks held it the diagonal was 2 and the residual 2 (u - g).

    One rank is made the authority for each shared dof that is Dirichlet anywhere:
    the owner if its dictionary holds the dof, otherwise the lowest rank that does.
    Every other copy (held or not) is dropped, so the assembled row is one identity
    and one ``u - g`` whatever the partition. No Dirichlet value is exchanged, so
    values that depend on x or t work. Collective, once per level model.

    Returns the local dof indices to zero (an int array, maybe empty).
    """
    mpi = Comm.get().comm.tompi4py()
    size, rank = mpi.Get_size(), mpi.Get_rank()
    if size == 1:
        return numpy.zeros(0, dtype=cmeshTools.INDEX_DTYPE)
    nproc = par_r.dim_proc
    s2g = numpy.asarray(par_r.subdomain2global)
    nloc = s2g.shape[0]
    hold = numpy.zeros(nloc, dtype=bool)
    outside = 0
    dbcs = model.dirichletConditionsForceDOF
    if hasattr(dbcs, 'DOFBoundaryConditionsDict'):  # one object, not one per component (a single-component model)
        dbcs = {0: dbcs}
    for cj, dbc in dbcs.items():
        for dofN in dbc.DOFBoundaryConditionsDict:
            d = model.offset[cj] + model.stride[cj] * dofN
            if d < nloc:
                hold[d] = True
            else:
                outside += 1
    # a layout whose Dirichlet dofs are not subdomain dofs (a serial-periodic numbering) is left as it was;
    # every rank must agree, so the decision is collective
    # (and likewise where two local dofs share a global number, as merged periodic dofs do)
    shared_numbers = int(numpy.unique(s2g).shape[0] != nloc)
    if mpi.allreduce(outside + shared_numbers, op=MPI.SUM) > 0:
        return numpy.zeros(0, dtype=cmeshTools.INDEX_DTYPE)
    ranges = numpy.asarray(par_r.getOwnershipRanges())
    ghosts = numpy.arange(nproc, nloc)
    gown = numpy.searchsorted(ranges, s2g[ghosts], side='right') - 1
    send = [None] * size
    for r in numpy.unique(gown):
        sel = gown == r
        send[r] = (s2g[ghosts[sel]], hold[ghosts[sel]])
    recv = mpi.alltoall(send)
    owned_order = numpy.argsort(s2g[:nproc])
    owned_sorted = s2g[:nproc][owned_order]
    # owner: per shared dof, who holds it
    any_holder = {}
    for src, item in enumerate(recv):
        if item is None:
            continue
        for g, h in zip(*item):
            rec = any_holder.setdefault(int(g), [None, None])  # [lowest holder rank among ghosts, owner holds]
            if h and (rec[0] is None or src < rec[0]):
                rec[0] = src
    zero_owned = []
    for g, rec in any_holder.items():
        d = int(owned_order[numpy.searchsorted(owned_sorted, g)])
        owner_holds = bool(hold[d])
        rec[1] = owner_holds
        if rec[0] is None and not owner_holds:
            continue  # nobody holds it
        authority = rank if owner_holds else rec[0]
        rec.append(authority)
        if authority != rank:
            zero_owned.append(d)
    reply = [None] * size
    for src, item in enumerate(recv):
        if item is None:
            continue
        flags = numpy.zeros(len(item[0]), dtype=bool)
        for k, g in enumerate(item[0]):
            rec = any_holder[int(g)]
            if len(rec) == 3 and rec[2] != src:
                flags[k] = True
        reply[src] = flags
    back = mpi.alltoall(reply)
    zero_ghost = []
    for r in numpy.unique(gown):
        sel = numpy.where(gown == r)[0]
        zero_ghost.extend(ghosts[sel][back[r]].tolist())
    # an owned dof that no other rank sees is never shared: nothing to do for it
    return numpy.array(sorted(zero_owned + zero_ghost), dtype=cmeshTools.INDEX_DTYPE)


def attachStrongDirichletAuthority(model, par_r, par_jacobian):
    """Install strongDirichletNonAuthorityDofs on the parallel residual vector and matrix."""
    if par_r is None or not getattr(model, 'forceStrongConditions', False):
        return
    if not (hasattr(model.dirichletConditionsForceDOF, 'items') or model.nc == 1):
        return
    if getattr(par_r, 'proteus2petsc_subdomain', None) is not None or par_r.getBlockSize() != 1:
        return  # permuted or block layouts are not handled
    idx = strongDirichletNonAuthorityDofs(model, par_r)
    par_r.dirichlet_nonauthority = idx
    par_jacobian.dirichlet_nonauthority = idx
    logEvent("Strong Dirichlet authority on rank %i: %i local dofs not contributed" % (Comm.get().rank(), idx.shape[0]), level=4)
