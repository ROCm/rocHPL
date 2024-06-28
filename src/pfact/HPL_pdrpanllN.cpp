/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

void HPL_pdrpanllN(HPL_T_panel* PANEL,
                   const int    M,
                   const int    N,
                   const int    ICOFF,
                   double*      WORK,
                   int          thread_rank,
                   int          thread_size,
                   double*      max_value,
                   int*         max_index) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdrpanllN recursively  factorizes  a panel  of columns using  the
   * recursive Left-looking variant of the one-dimensional algorithm.  The
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
   * words,  and  gam2-3  is an estimate of the  Level 2 and Level 3  BLAS
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
   * WORK    (local workspace)             double *
   *         On entry, WORK  is a workarray of size at least 2*(4+2*N0).
   *
   * ---------------------------------------------------------------------
   */

  double *A, *Aptr, *L1, *L1ptr;
  int     curr, ii, ioff, jb, jj, lda, m, n, n0, nb, nbdiv, nbmin;

  if(N <= (nbmin = PANEL->algo->nbmin)) {
    PANEL->algo->pffun(PANEL,
                       M,
                       N,
                       ICOFF,
                       WORK,
                       thread_rank,
                       thread_size,
                       max_value,
                       max_index);
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

  A     = PANEL->A;
  lda   = PANEL->lda;
  L1    = PANEL->L1;
  n0    = PANEL->jb;
  L1ptr = Mptr(L1, ICOFF, ICOFF, n0);
  curr  = (int)(PANEL->grid->myrow == PANEL->prow);

  if(curr != 0)
    Aptr = Mptr(A, ICOFF, ICOFF, lda);
  else
    Aptr = Mptr(A, 0, ICOFF, lda);
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
     * Replicated solve - Local update - Factor current panel
     */
    if(thread_rank == 0) {
      HPL_dtrsm(HplColumnMajor,
                HplLeft,
                HplLower,
                HplNoTrans,
                HplUnit,
                jj,
                jb,
                HPL_rone,
                L1ptr,
                n0,
                Mptr(L1ptr, 0, jj, n0),
                n0);
    }

   #pragma omp barrier

    HPL_dgemm_omp(HplColumnMajor,
                  HplNoTrans,
                  HplNoTrans,
                  m,
                  jb,
                  jj,
                  -HPL_rone,
                  Mptr(Aptr, ii, 0, lda),
                  lda,
                  Mptr(L1ptr, 0, jj, n0),
                  n0,
                  HPL_rone,
                  Mptr(Aptr, ii, jj, lda),
                  lda,
                  PANEL->nb,
                  (curr != 0) ? ICOFF + ii : 0,
                  thread_rank,
                  thread_size);

    HPL_pdrpanllN(PANEL,
                  m,
                  jb,
                  ioff,
                  WORK,
                  thread_rank,
                  thread_size,
                  max_value,
                  max_index);
    /*
     * Copy back upper part of A in current process row - Go the next block
     */
    if(curr != 0) {
      if(thread_rank == 0) {
        HPL_dlacpy(
            ioff, jb, Mptr(L1, 0, ioff, n0), n0, Mptr(A, 0, ioff, lda), lda);
      }
      ii += jb;
      m -= jb;
    }
    jj += jb;
    jb = Mmin(n, nb);

  } while(n > 0);
}
