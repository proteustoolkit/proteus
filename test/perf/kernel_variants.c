/* What bounds femIntegrals' updateDiffusion_weak_sd? The kernel in variants.
 *
 * The legacy element kernels run at 2-6% of a core's bandwidth and a few percent of its
 * FMA rate (ON_NODE_PERFORMANCE_BASELINE.md). That says what they are not bound by; this
 * says what they are. The residual pass of the sparse diffusion term
 *
 *   weak_residual[eN,i] += sum_k sum_I sum_m a[eN,k,m] grad_phi[eN,k,colind[m]] grad_w_dV[eN,k,i,I]
 *
 * is compiled five ways from the same inputs:
 *
 *   v0 original   verbatim from femIntegrals.c: the sum accumulates into weak_residual in
 *                 memory and every operand's flat index is recomputed from the loop variables
 *   v1 local      the sum over (k, I, m) accumulates in a register
 *   v2 hoisted    v1 plus the index arithmetic hoisted into running pointers
 *   v3 constant   v2 with the element's shape (points, dofs, space, nonzeros) known at compile
 *                 time, as a generated kernel has it
 *   v4 diagonal   v3 with the tensor's zero entries removed at compile time: 3 terms, not 9
 *
 * v0-v3 give the same bits (the order of the sums is unchanged); v4 differs by the terms that
 * are zero in the test data, and is checked against the others to round-off.
 *
 * Usage: kernel_variants [nElements] [repeats] [variant]
 * With a variant (0-4) only that one runs, for `perf stat`. Prints one line per variant:
 * ns per element, ns per innermost iteration of v0 (the same iteration count for all, so the
 * numbers compare), GFLOP/s of the 3-flop-per-iteration count v0-v3 share.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NQP 8
#define NDOF 8
#define NSPACE 3
#define NNZ 9

typedef void (*kernel_t)(int, int, int, int, int*, int*, double*, double*, double*, double*);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* v0: femIntegrals.c, updateDiffusion_weak_sd */
void v0(int nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace,
        int* rowptr, int* colind, double* a, double* grad_phi, double* grad_w_dV, double* weak_residual)
{
  int eN,i,k,I,m,nnz=rowptr[nSpace];
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      for (k=0;k<nQuadraturePoints_element;k++)
        for (I=0;I<nSpace;I++)
          for (m=rowptr[I];m<rowptr[I+1];m++)
            weak_residual[eN*nDOF_test_element + i]
              +=
              a[eN*nQuadraturePoints_element*nnz+ k*nnz + m]
              *
              grad_phi[eN*nQuadraturePoints_element*nSpace + k*nSpace + colind[m]]
              *
              grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace
                        + k*nDOF_test_element*nSpace + i*nSpace + I];
}

/* v1: the sum in a register */
void v1(int nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace,
        int* rowptr, int* colind, double* a, double* grad_phi, double* grad_w_dV, double* weak_residual)
{
  int eN,i,k,I,m,nnz=rowptr[nSpace];
  for(eN=0;eN<nElements_global;eN++)
    for (i=0;i<nDOF_test_element;i++)
      {
        double sum = weak_residual[eN*nDOF_test_element + i];
        for (k=0;k<nQuadraturePoints_element;k++)
          for (I=0;I<nSpace;I++)
            for (m=rowptr[I];m<rowptr[I+1];m++)
              sum += a[eN*nQuadraturePoints_element*nnz+ k*nnz + m]
                *
                grad_phi[eN*nQuadraturePoints_element*nSpace + k*nSpace + colind[m]]
                *
                grad_w_dV[eN*nQuadraturePoints_element*nDOF_test_element*nSpace
                          + k*nDOF_test_element*nSpace + i*nSpace + I];
        weak_residual[eN*nDOF_test_element + i] = sum;
      }
}

/* v2: v1 with the index arithmetic hoisted into running pointers */
void v2(int nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace,
        int* rowptr, int* colind, double* a, double* grad_phi, double* grad_w_dV, double* weak_residual)
{
  int eN,i,k,I,m,nnz=rowptr[nSpace];
  for(eN=0;eN<nElements_global;eN++)
    {
      const double* ae = a + (long)eN*nQuadraturePoints_element*nnz;
      const double* gp = grad_phi + (long)eN*nQuadraturePoints_element*nSpace;
      const double* gw = grad_w_dV + (long)eN*nQuadraturePoints_element*nDOF_test_element*nSpace;
      for (i=0;i<nDOF_test_element;i++)
        {
          double sum = weak_residual[eN*nDOF_test_element + i];
          const double* ak = ae;
          const double* gpk = gp;
          const double* gwk = gw + i*nSpace;
          for (k=0;k<nQuadraturePoints_element;k++)
            {
              for (I=0;I<nSpace;I++)
                for (m=rowptr[I];m<rowptr[I+1];m++)
                  sum += ak[m] * gpk[colind[m]] * gwk[I];
              ak += nnz; gpk += nSpace; gwk += nDOF_test_element*nSpace;
            }
          weak_residual[eN*nDOF_test_element + i] = sum;
        }
    }
}

/* v3: v2 with the shape known at compile time */
void v3(int nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace,
        int* rowptr, int* colind, double* a, double* grad_phi, double* grad_w_dV, double* weak_residual)
{
  for(int eN=0;eN<nElements_global;eN++)
    {
      const double* ae = a + (long)eN*NQP*NNZ;
      const double* gp = grad_phi + (long)eN*NQP*NSPACE;
      const double* gw = grad_w_dV + (long)eN*NQP*NDOF*NSPACE;
      for (int i=0;i<NDOF;i++)
        {
          double sum = weak_residual[eN*NDOF + i];
          for (int k=0;k<NQP;k++)
            for (int I=0;I<NSPACE;I++)
              for (int m=rowptr[I];m<rowptr[I+1];m++)
                sum += ae[k*NNZ+m] * gp[k*NSPACE+colind[m]] * gw[(k*NDOF+i)*NSPACE+I];
          weak_residual[eN*NDOF + i] = sum;
        }
    }
}

/* v4: v3 with the tensor's structure known at compile time too: diagonal, 3 terms */
void v4(int nElements_global, int nQuadraturePoints_element, int nDOF_test_element, int nSpace,
        int* rowptr, int* colind, double* a, double* grad_phi, double* grad_w_dV, double* weak_residual)
{
  for(int eN=0;eN<nElements_global;eN++)
    {
      const double* ae = a + (long)eN*NQP*NNZ;
      const double* gp = grad_phi + (long)eN*NQP*NSPACE;
      const double* gw = grad_w_dV + (long)eN*NQP*NDOF*NSPACE;
      for (int i=0;i<NDOF;i++)
        {
          double sum = weak_residual[eN*NDOF + i];
          for (int k=0;k<NQP;k++)
            for (int I=0;I<NSPACE;I++)
              sum += ae[k*NNZ+I*NSPACE+I] * gp[k*NSPACE+I] * gw[(k*NDOF+i)*NSPACE+I];
          weak_residual[eN*NDOF + i] = sum;
        }
    }
}

static kernel_t kernels[] = {v0, v1, v2, v3, v4};
static const char* names[] = {"v0 original", "v1 local sum", "v2 hoisted index", "v3 constant shape",
                              "v4 diagonal terms"};

int main(int argc, char** argv)
{
  int n = argc > 1 ? atoi(argv[1]) : 8000;
  int repeats = argc > 2 ? atoi(argv[2]) : 20;
  int only = argc > 3 ? atoi(argv[3]) : -1;
  int rowptr[4] = {0, 3, 6, 9};
  int colind[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};      /* the full tensor, as Poisson passes it */
  long na = (long)n*NQP*NNZ, ng = (long)n*NQP*NSPACE, nw = (long)n*NQP*NDOF*NSPACE, nr = (long)n*NDOF;
  double *a = malloc(na*8), *gp = malloc(ng*8), *gw = malloc(nw*8), *r = malloc(nr*8), *ref = malloc(nr*8);
  srand(7);
  for (long j = 0; j < na; j++) a[j] = 0.5 + (double)rand()/RAND_MAX;
  /* a diagonal tensor: off-diagonal entries are zero, as in the heterogeneous Poisson test */
  for (int e = 0; e < n*NQP; e++)
    for (int I = 0; I < NSPACE; I++)
      for (int J = 0; J < NSPACE; J++)
        if (I != J) a[(long)e*NNZ + I*NSPACE + J] = 0.0;
  for (long j = 0; j < ng; j++) gp[j] = (double)rand()/RAND_MAX - 0.5;
  for (long j = 0; j < nw; j++) gw[j] = (double)rand()/RAND_MAX - 0.5;
  double iterations = (double)n * NDOF * NQP * NNZ;       /* innermost iterations of v0..v3 */
  if (only > 0)                                           /* the reference, unreported */
    {
      memset(ref, 0, nr*8);
      kernels[0](n, NQP, NDOF, NSPACE, rowptr, colind, a, gp, gw, ref);
    }
  for (int v = 0; v < 5; v++)
    {
      if (only >= 0 && v != only) continue;
      double best = 1e30;
      for (int rep = 0; rep < repeats; rep++)
        {
          memset(r, 0, nr*8);
          double t = now();
          kernels[v](n, NQP, NDOF, NSPACE, rowptr, colind, a, gp, gw, r);
          t = now() - t;
          if (t < best) best = t;
        }
      if (v == 0 && only <= 0) memcpy(ref, r, nr*8);
      double worst = 0.0;
      for (long j = 0; j < nr; j++) worst = fmax(worst, fabs(r[j] - ref[j]));
      printf("%-18s n=%-7d %8.1f ns/element %6.3f ns/iteration  %5.2f GFLOP/s (3 flop/iteration)  max|diff from v0| %.1e\n",
             names[v], n, 1e9*best/n, 1e9*best/iterations, 3.0*iterations/best/1e9, worst);
    }
  return 0;
}
