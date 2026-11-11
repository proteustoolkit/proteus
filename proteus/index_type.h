#ifndef PROTEUS_INDEX_TYPE_H
#define PROTEUS_INDEX_TYPE_H
#include <stdint.h>

/**
   \file index_type.h
   \brief The integer type of mesh entity numbers, counts and offsets.

   One definition read by both sides: mesh.h declares struct Mesh with it, and
   cmeshTools exports its width to Python as cmeshTools.INDEX_DTYPE, which is
   the dtype of every index array a CMesh hands to numpy.

   Changing the width is not yet a one-line change: the kernels and the Python
   modules still say int and 'i'. Changing it here makes every place that
   disagrees a compile error, or a failure in test/test_index_type.py.

   Local numbers with a small fixed range (an element's own faces, material
   tags) are a different type and stay int.
*/
typedef int32_t proteus_index_t;

/**
   \brief The integer type of the arrays that describe a CSR matrix: row
   pointers, column indices, and the offsets into nzval that the assembly
   kernels precompute (csrRowIndeces, csrColumnOffsets*).

   Decided 2026-11-10 (C33): this is its own type, equal to PetscInt, and it
   varies independently of proteus_index_t: a mesh with 3e9 nodes and a
   32-bit-PETSc matrix are different questions. setup.py reads PETSc's
   configuration and passes -DPROTEUS_CSR_INDEX_BITS=32 or 64; without it the
   type is 32 bits, the width of every PETSc and SuperLU build proteus ships.
   Files that include petsc headers use PROTEUS_CSR_INDEX_MATCHES_PETSC to
   static_assert the match. The SuperLU wrappers keep SuperLU's own int and
   convert only when the widths differ (superluWrappers.pyx).
*/
#ifndef PROTEUS_CSR_INDEX_BITS
#define PROTEUS_CSR_INDEX_BITS 32
#endif
#if PROTEUS_CSR_INDEX_BITS == 64
typedef int64_t proteus_csr_index_t;
#elif PROTEUS_CSR_INDEX_BITS == 32
typedef int32_t proteus_csr_index_t;
#else
#error "PROTEUS_CSR_INDEX_BITS must be 32 or 64"
#endif
#ifdef __cplusplus
#define PROTEUS_CSR_INDEX_MATCHES_PETSC static_assert(sizeof(proteus_csr_index_t) == sizeof(PetscInt), "proteus_csr_index_t must equal PetscInt: rebuild with the PROTEUS_CSR_INDEX_BITS that setup.py derives from PETSc")
#endif

#endif
