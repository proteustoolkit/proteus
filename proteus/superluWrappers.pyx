#cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

# ARB - the cython directives above allow code to run much faster.  it is worth noting
# however that these directives can lead to segfaults if the code is not implemented
# correctly.  perhaps they should only be enabled when not in debug mode?

import numpy as np
import cython
cimport numpy as np

from libc.stdlib cimport malloc

class SparseMatrix(object):

    def __init__(self,
                  nr,
                  nc,
                  nnz,
                  nzvals,
                  colind,
                  rowptr):
        self.nr = nr ; self.nc = nc
        self.nnz = nnz
        self.shape = [self.nr, self.nc]
        self.nzvals = nzvals
        self.colind = colind
        self.rowptr = rowptr
        self._cSparseMatrix = cSparseMatrix(nr,
                                            nc,
                                            nnz,
                                            self.nzvals,
                                            self.colind,
                                            self.rowptr)

    def matvec(self, x, y):
        """
        Compute the sparse matrix-vector product y = Ax

        Arguments
        ---------
        x (input) :  numpy array
        y (output) : numpy array
        """
        # SparseMatrix_matvec is compiled with boundscheck=False, so a matrix
        # larger than the vectors it is handed writes past the end of y and
        # reads past the end of x instead of raising -- silent heap corruption
        # that surfaces much later, in an unrelated allocation. The dimensions
        # here are exactly the precondition the loop needs, and checking them
        # is O(1) against the O(nnz) product.
        if self.nc > len(x) or self.nr > len(y):
            raise ValueError(
                "matvec: matrix is %ix%i but x has %i and y has %i entries"
                % (self.nr, self.nc, len(x), len(y)))
        SparseMatrix_matvec(self._cSparseMatrix, x, y)

    def fwrite(self, filename, base):
        """ Write the sparse matrix to a file

        Arguments
        ---------
        filename : str
            The output filename
        base : int
            ?!Possibly something to do with parallel?!
        """
        with open(filename, 'w') as output_file:
            output_file.write('%i %i %i \n' % (self.nr, self.nc, self.rowptr[self.nr]) )
            for i in range(self.nr):
                for k in range(self.rowptr[i], self.rowptr[i+1]):
                    output_file.write('%d %d %13.8e\n' % (i+base, self.colind[k]+base, self.nzvals[k]) )

    def getCSRrepresentation(self):
        """ Get the CSR representation of the sparse matrix.

        Returns
        -------
        csr_data : tuple of nparray
            (rowptr, colptr, vals)
        """
        return (self.rowptr, self.colind, self.nzvals)

    def getSubMatCSRrepresentation(self,
                                   range_start,
                                   range_end):
        """  Get the CSR representation for a submatrix.

        Arguments
        ---------
        range_start : int
        range_end : int

        Returns
        -------
        csr_data : tuple of nparray
            (rowptr, colind, nzvals)
        """
        assert range_start >= 0 ; assert range_end <= self.nr
        #an empty range is legitimate: a rank can own no rows of a small
        #(coarse) level, and still has to take part in assembling it
        assert range_end >= range_start
        _rows = range_end - range_start
        assert _rows <= self.nr

        nnz = self.rowptr[range_end] - self.rowptr[range_start]

        rowptr = self.rowptr[range_start : range_start + _rows + 1]
        colind = self.colind[self.rowptr[range_start] : self.rowptr[range_start] + nnz]
        nzvals = self.nzvals[self.rowptr[range_start] : self.rowptr[range_start] + nnz]

        return rowptr, colind, nzvals

cdef class cSparseMatrix(object):

    def __cinit__(self,
                 int nr,
                 int nc,
                 int nnz,
                 np.float64_t [:] nzval,
                 object colind,
                 object rowptr):
        # colind and rowptr arrive in the CSR index type (csparsity.CSR_INDEX_DTYPE,
        # = PetscInt) and SuperLU takes int: the structure is converted here, and
        # only when the widths differ (ascontiguousarray returns the caller's array
        # when the dtype already matches, so the usual build shares memory as before).
        if nnz > 2147483647 or max(nr, nc) > 2147483647:
            raise OverflowError("superluWrappers: a %i x %i matrix with %i nonzeros does not fit SuperLU's 32-bit int" % (nr, nc, nnz))
        self._colind32 = np.ascontiguousarray(colind, dtype=np.int32)
        self._rowptr32 = np.ascontiguousarray(rowptr, dtype=np.int32)
        cdef np.int32_t [:] c32 = self._colind32
        cdef np.int32_t [:] r32 = self._rowptr32
        self.dim[0] = nr ; self.dim[1] = nc
        self.A.nnz = nnz
        #ARB - should memory for these ptrs need be allocated?
        self.A.nzval = &nzval[0]
        self.A.colind = &c32[0]
        self.A.rowptr = &r32[0]

cdef void SparseMatrix_matvec(cSparseMatrix sm,
                              np.float64_t [:] xp,
                              np.float64_t [:] yp):
    cdef np.float64_t tmp = 0.
    cdef int i, k
    cdef int k_lo, k_hi

    for i in range(sm.dim[0]):
        tmp = 0.
        k_lo = sm.A.rowptr[i]
        k_hi = sm.A.rowptr[i+1]
        for k in range(k_lo, k_hi):
            tmp += sm.A.nzval[k] * xp[sm.A.colind[k]]
        yp[i] = tmp

cdef struct _NCformat:
    np.int32_t nnz
    np.float64_t * nzval
    np.int32_t * rowind
    np.int32_t * colptr

cdef struct _DNformat:
    np.int32_t lda
    void *nzval

cdef struct _NCPformat:
    np.int32_t nnz
    void * nzval
    np.int32_t * rowind
    np.int32_t * colbeg
    np.int32_t * colend

cdef class SparseFactor(object):

    cdef _superlu_options_t options

    cdef _SuperMatrix A
    cdef _SuperMatrix AC
    cdef _SuperMatrix L
    cdef _SuperMatrix U
    cdef _SuperMatrix X

    cdef _GlobalLU_t Glu
    cdef _SuperLUStat_t stat

    cdef _NCformat storeA
    cdef _DNformat storeX

    cdef int * perm_c
    cdef int * perm_r
    cdef int * etree

    cdef unsigned int use_same_perm_c
    cdef unsigned int use_same_sparsity

    cdef public int dim

    def __init__(self, dim):
        """
        Arguments
        ---------
        dim : int
            Dimension of the sparse factor.
        """
        cStatInit(&self.stat)
        cset_default_options(&self.options)
        self._set_mat_types()
        self.dim = dim
        self.A.nrow = dim ; self.A.ncol = dim
        self.AC.nrow = dim ; self.AC.ncol = dim
        self.L.nrow = dim ; self.L.ncol = dim
        self.U.nrow = dim ; self.U.ncol = dim
        self.X.nrow = dim ; self.X.ncol = 1
        self.storeX.lda = dim
        self.use_same_perm_c = 0
        self.use_same_sparsity = 0
        self.perm_c = <int *>malloc(dim*sizeof(np.int32_t))
        self.perm_r = <int *>malloc(dim*sizeof(np.int32_t))
        self.etree = <int *>malloc(dim*sizeof(np.int32_t))

    cdef _set_mat_types(self):
        self.A.Stype = _SLU_NC
        self.A.Dtype = _SLU_D
        self.A.Mtype = _SLU_GE
        self.A.Store = &self.storeA

        self.AC.Stype = _SLU_NCP
        self.AC.Dtype = _SLU_D
        self.AC.Mtype = _SLU_GE
        self.AC.Store = NULL

        self.L.Stype = _SLU_NC
        self.L.Dtype = _SLU_D
        self.L.Mtype = _SLU_TRLU
        self.L.Store = NULL

        self.U.Stype = _SLU_NC
        self.U.Dtype = _SLU_D
        self.U.Mtype = _SLU_TRU
        self.U.Store = NULL

        self.X.Stype = _SLU_DN
        self.X.Dtype = _SLU_D
        self.X.Mtype = _SLU_GE
        self.X.Store = &self.storeX

def sparseFactorPrepare(sparse_matrix,
                        sparseFactor):
    """ Python wrapper for superlu Sparse Factor Prepare function.

    Arguments
    ---------
    sparse_matrix : superluWrappers.SparseMatrix
    sparseFactor: superluWrappers.SparseFactor

    """
    superluWrappersSparseFactorPrepare(sparse_matrix._cSparseMatrix,
                                       sparseFactor)

cdef void superluWrappersSparseFactorPrepare(cSparseMatrix sm,
                                             SparseFactor sparseFactor):

    cdef int permc_spec = 3
    cdef int n
    cdef int relax=1
    cdef int panel_size = 10
    cdef int lwork = 0
    cdef int info = 0
    cdef void *work = NULL

    sparseFactor.storeA.nnz = sm.A.nnz
    sparseFactor.storeA.nzval = &sm.A.nzval[0]
    sparseFactor.storeA.colptr = &sm.A.rowptr[0]
    sparseFactor.storeA.rowind = &sm.A.colind[0]

    if sparseFactor.use_same_perm_c == 0:
        cget_perm_c(permc_spec,
                    &sparseFactor.A,
                    sparseFactor.perm_c)
        sparseFactor.use_same_perm_c = 1

    if sparseFactor.use_same_sparsity == 0:
        if sparseFactor.AC.Store != NULL:
            cDestroy_CompCol_Permuted(&sparseFactor.AC)
            cDestroy_SuperNode_Matrix(&sparseFactor.L)
            cDestroy_CompCol_Matrix(&sparseFactor.U)
        csp_preorder(&sparseFactor.options,
                     &sparseFactor.A,
                     sparseFactor.perm_c,
                     sparseFactor.etree,
                     &sparseFactor.AC)
        sparseFactor.use_same_sparsity = 1
    else:
        sparseFactor.options.Fact = _SamePattern_SameRowPerm
        n = sparseFactor.A.ncol
        for i in range(n):
            (<_NCPformat *>sparseFactor.AC.Store).colbeg[sparseFactor.perm_c[i]] = (<_NCformat *> sparseFactor.A.Store).colptr[i]
            (<_NCPformat *>sparseFactor.AC.Store).colend[sparseFactor.perm_c[i]] = (<_NCformat *> sparseFactor.A.Store).colptr[i+1]
    cdgstrf(&sparseFactor.options,
            &sparseFactor.AC,
            relax,
            panel_size,
            sparseFactor.etree,
            work,
            lwork,
            sparseFactor.perm_c,
            sparseFactor.perm_r,
            &sparseFactor.L,
            &sparseFactor.U,
            &sparseFactor.Glu,
            &sparseFactor.stat,
            &info)

def sparseFactorSolve(sparseFactor,
                      x):
    """  Sparse factor solve wrappers

    Arguments
    ---------
    sparseFactor : superluWrappers.SparseFactor
    x (input / output) : np.array
       x serves as the right hand side and then becomes the solution
    """
    superluWrappersSparseFactorSolve(sparseFactor,
                                     x)


cdef void superluWrappersSparseFactorSolve(SparseFactor sparseFactor,
                                           np.float64_t [:] x):
    cdef _trans_t trans = _TRANS
    cdef int info = 0

    sparseFactor.storeX.nzval = &x[0]

    cdgstrs(trans,
            &sparseFactor.L,
            &sparseFactor.U,
            sparseFactor.perm_c,
            sparseFactor.perm_r,
            &sparseFactor.X,
            &sparseFactor.stat,
            &info)
