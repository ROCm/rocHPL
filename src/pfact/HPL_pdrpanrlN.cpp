/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

void HPL_pdrpanrlN(HPL_T_panel* PANEL,
                   const int    M,
                   const int    N,
                   const int    ICOFF) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdrpanrlN recursively  factorizes  a panel of columns  using  the
   * recursive Right-looking variant of the one-dimensional algorithm. The
   * lower triangular  N0-by-N0  upper block  of  the  panel  is stored in
   * no-transpose form (i.e. just like the input matrix itself).
   *
   * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
   * operations  at once  for one column in the panel.  This  results in a
   * lower number of slightly larger  messages than usual.  On P processes
   * and assuming bi-directional links,  the running time of this function
   * can be approximated by (when N is equal to N0):
   *
   *    N0 * log_2( P ) * ( lat + ( 2*N0 + 4 ) / bdwth ) +
   *    N0^2 * ( M - N0/3 ) * gam2-3
   *
   * where M is the local number of rows of  the panel, lat and bdwth  are
   * the latency and bandwidth of the network for  double  precision  real
   * words, and   gam2-3  is an estimate of the  Level 2 and Level 3  BLAS
   * rate of execution. The  recursive  algorithm  allows indeed to almost
   * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
   * large  number of modern machines,  this  operation is however latency
   * bound,  meaning  that its cost can  be estimated  by only the latency
   * portion N0 * log_2(P) * lat.  Mono-directional links will double this
   * communication cost.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the local number of rows of sub(A).
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the local number of columns of sub(A).
   *
   * ICOFF   (global input)                const int
   *         On entry, ICOFF specifies the row and column offset of sub(A)
   *         in A.
   *
   * ---------------------------------------------------------------------
   */

  double *A, *Aptr, *L1, *L1ptr;
  int     curr, ii, ioff, jb, jj, lda, m, n, n0, nb, nbdiv, nbmin;

  if(N <= (nbmin = PANEL->algo->nbmin)) {
    PANEL->algo->pffun(PANEL, M, N, ICOFF);
    return;
  }
  /*
   * Find  new recursive blocking factor.  To avoid an infinite loop,  one
   * must guarantee: 1 <= jb < N, knowing that  N  is greater than  NBMIN.
   * First, we compute nblocks:  the number of blocks of size  NBMIN in N,
   * including the last one that may be smaller.  nblocks  is thus  larger
   * than or equal to one, since N >= NBMIN.
   * The ratio ( nblocks + NDIV - 1 ) / NDIV  is thus larger than or equal
   * to one as  well.  For  NDIV >= 2,  we  are guaranteed  that the quan-
   * tity ( ( nblocks + NDIV  - 1 ) / NDIV ) * NBMIN  is less  than N  and
   * greater than or equal to NBMIN.
   */
  nbdiv = PANEL->algo->nbdiv;
  ii = jj = 0;
  m       = M;
  n       = N;
  nb = jb = ((((N + nbmin - 1) / nbmin) + nbdiv - 1) / nbdiv) * nbmin;

  A     = PANEL->A0;
  lda   = PANEL->lda0;
  L1    = PANEL->L1;
  n0    = PANEL->jb;
  L1ptr = Mptr(L1, ICOFF, ICOFF, n0);
  curr  = (int)(PANEL->grid->myrow == PANEL->prow);

  if(curr != 0)
    Aptr = Mptr(A, ICOFF, ICOFF, lda);
  else
    Aptr = Mptr(A, 0, ICOFF, lda);

  const double one  = 1.0;
  const double mone = -1.0;
  /*
   * The triangular solve is replicated in every  process row.  The  panel
   * factorization is  such that  the first rows of  A  are accumulated in
   * every process row during the (panel) swapping phase.  We  ensure this
   * way a minimum amount  of communication during the entire panel facto-
   * rization.
   */
  do {
    n -= jb;
    ioff = ICOFF + jj;
    /*
     * Factor current panel - Replicated solve - Local update
     */
    HPL_pdrpanrlN(PANEL, m, jb, ioff);

    CHECK_ROCBLAS_ERROR(rocblas_dtrsm(handle,
                                      rocblas_side_left,
                                      rocblas_fill_lower,
                                      rocblas_operation_none,
                                      rocblas_diagonal_unit,
                                      jb,
                                      n,
                                      &one,
                                      Mptr(L1ptr, jj, jj, n0),
                                      n0,
                                      Mptr(L1ptr, jj, jj + jb, n0),
                                      n0));

    if(curr != 0) {
      ii += jb;
      m -= jb;
    }

    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_none,
                                      m,
                                      n,
                                      jb,
                                      &mone,
                                      Mptr(Aptr, ii, jj, lda),
                                      lda,
                                      Mptr(L1ptr, jj, jj + jb, n0),
                                      n0,
                                      &one,
                                      Mptr(Aptr, ii, jj + jb, lda),
                                      lda));

    jj += jb;
    jb = Mmin(n, nb);

  } while(n > 0);
}
