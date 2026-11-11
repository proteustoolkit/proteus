# cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "proteus_superlu.h":
    ctypedef enum _Stype_t 'Stype_t':
        _SLU_NC 'SLU_NC'
        _SLU_NCP 'SLU_NCP'
        _SLU_NR 'SLU_NR'
        _SLU_SC 'SLU_SC'
        _SLU_SCP 'SLU_SCP'
        _SLU_SR 'SLU_SR'
        _SLU_DN 'SLU_DN'
        _SLU_NR_loc 'SLU_NR_loc'
    ctypedef enum _Dtype_t 'Dtype_t':
        _SLU_S 'SLU_S'
        _SLU_D 'SLU_D'
        _SLU_C 'SLU_C'
        _SLU_Z 'SLU_Z'
    ctypedef enum _Mtype_t 'Mtype_t':
        _SLU_GE 'SLU_GE'
        _SLU_TRLU 'SLU_TRLU'
        _SLU_TRUU 'SLU_TRUU'
        _SLU_TRL 'SLU_TRL'
        _SLU_TRU 'SLU_TRU'
        _SLU_SYL 'SLU_SYL'
        _SLU_SYU 'SLU_SYU'
        _SLU_HEL 'SLU_HEL'
        _SLU_HEU 'SLU_HEU'
    ctypedef struct _SuperMatrix 'SuperMatrix':
        _Stype_t Stype
        _Dtype_t Dtype
        _Mtype_t Mtype
        int nrow
        int ncol
        void * Store
    ctypedef enum _fact_t 'fact_t':
        _DOFACT 'DFACT'
        _SamePattern "SamePattern"
        _SamePattern_SameRowPerm "SamePattern_SameRowPerm"
        _FACTORED "FACTORED"
    ctypedef enum _trans_t "trans_t":
        _NOTRANS 'NOTRANS'
        _TRANS 'TRANS'
        _CONJ 'CONJ'
    ctypedef struct _SuperLUStat_t 'SuperLUStat_t':
        pass
    ctypedef struct _GlobalLU_t "GlobalLU_t":
        pass
    ctypedef struct _superlu_options_t "superlu_options_t":
        _fact_t Fact
    void cdgstrs "dgstrs"(_trans_t, _SuperMatrix *, _SuperMatrix *, int *, int *, _SuperMatrix *, _SuperLUStat_t*, int *)
    void cdgstrf "dgstrf"(_superlu_options_t *, _SuperMatrix *, int, int, int *, void *, int, int *, int *, _SuperMatrix *, _SuperMatrix *, _GlobalLU_t*, _SuperLUStat_t *, int *)
    void cStatInit "StatInit"(_SuperLUStat_t *)
    void cset_default_options "set_default_options"(_superlu_options_t *)
    void cget_perm_c "get_perm_c"(int, _SuperMatrix *, int *)
    void cDestroy_CompCol_Permuted "Destroy_CompCol_Permuted"(_SuperMatrix *)
    void cDestroy_SuperNode_Matrix "Destroy_SuperNode_Matrix"(_SuperMatrix *)
    void cDestroy_CompCol_Matrix "Destroy_CompCol_Matrix"(_SuperMatrix *)
    void csp_preorder "sp_preorder"(_superlu_options_t *, _SuperMatrix *, int *, int *, _SuperMatrix *)

cdef struct _NRformat:
    np.int32_t nnz
    np.float64_t * nzval
    np.int32_t * colind
    np.int32_t * rowptr

cdef class cSparseMatrix(object):
    cdef np.int32_t dim[2]
    cdef object _colind32, _rowptr32  # SuperLU-width copies, or the caller's arrays when the widths agree
    cdef _NRformat A
