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

void HPL_pdpanllN(HPL_T_panel* PANEL,
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
   * HPL_pdpanllN factorizes  a panel of columns that is a sub-array of a
   * larger one-dimensional panel A  using the Left-looking variant of the
   * usual one-dimensional algorithm.  The lower triangular N0-by-N0 upper
   * block of the panel is stored in no-transpose form (i.e. just like the
   * input matrix itself).
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
   * words, and  gam2-3  is  an estimate of the  Level 2 and Level 3  BLAS
   * rate of execution. The  recursive  algorithm  allows indeed to almost
   * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
   * large  number of modern machines,  this  operation is however latency
   * bound,  meaning  that its cost can  be estimated  by only the latency
   * portion N0 * log_2(P) * lat.  Mono-directional links will double this
   * communication cost.
   *
   * Note that  one  iteration of the the main loop is unrolled. The local
   * computation of the absolute value max of the next column is performed
   * just after its update by the current column. This allows to bring the
   * current column only  once through  cache at each  step.  The  current
   * implementation  does not perform  any blocking  for  this sequence of
   * BLAS operations, however the design allows for plugging in an optimal
   * (machine-specific) specialized  BLAS-like kernel.  This idea has been
   * suggested to us by Fred Gustavson, IBM T.J. Watson Research Center.
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

  double *A, *L1, *L1ptr;
  int     Mm1, Nm1, curr, ii, iip1, jj, kk, lda, m = M, n0;
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  if(thread_rank == 0) HPL_ptimer(HPL_TIMING_PFACT);
#endif
  A    = PANEL->A0;
  lda  = PANEL->lda0;
  L1   = PANEL->L1;
  n0   = PANEL->jb;
  curr = (int)(PANEL->grid->myrow == PANEL->prow);

  Nm1 = N - 1;
  jj  = ICOFF;
  if(curr != 0) {
    ii   = ICOFF;
    iip1 = ii + 1;
    Mm1  = m - 1;
  } else {
    ii   = 0;
    iip1 = ii;
    Mm1  = m;
  }

  /*
   * Find local absolute value max in first column and initialize WORK[0:3]
   */
  HPL_dlocmax(
      PANEL, m, ii, jj, WORK, thread_rank, thread_size, max_index, max_value);

  while(Nm1 > 0) {
    /*
     * Swap and broadcast the current row
     */
    if(thread_rank == 0) {
      HPL_pdmxswp(PANEL, m, ii, jj, WORK);
      HPL_dlocswpN(PANEL, ii, jj, WORK);
    }

    L1ptr = Mptr(L1, ICOFF, jj + 1, n0);
    kk    = jj + 1 - ICOFF;
    if(thread_rank == 0) {
      HPL_dtrsv(HplColumnMajor,
                HplLower,
                HplNoTrans,
                HplUnit,
                kk,
                Mptr(L1, ICOFF, ICOFF, n0),
                n0,
                L1ptr,
                1);
    }

#pragma omp barrier

    /*
     * Scale  current column by its absolute value max entry  -  Update  and
     * find local  absolute value max  in next column (Only one pass through
     * cache for each next column).  This sequence of operations could bene-
     * fit from a specialized  blocked implementation.
     */
    if(WORK[0] != HPL_rzero)
      HPL_dscal_omp(Mm1,
                    HPL_rone / WORK[0],
                    Mptr(A, iip1, jj, lda),
                    1,
                    PANEL->nb,
                    iip1,
                    thread_rank,
                    thread_size);

    HPL_dgemv_omp(HplColumnMajor,
                  HplNoTrans,
                  Mm1,
                  kk,
                  -HPL_rone,
                  Mptr(A, iip1, ICOFF, lda),
                  lda,
                  L1ptr,
                  1,
                  HPL_rone,
                  Mptr(A, iip1, jj + 1, lda),
                  1,
                  PANEL->nb,
                  iip1,
                  thread_rank,
                  thread_size);

    HPL_dlocmax(PANEL,
                Mm1,
                iip1,
                jj + 1,
                WORK,
                thread_rank,
                thread_size,
                max_index,
                max_value);
    if(curr != 0) {
      if(thread_rank == 0) {
        HPL_dcopy(kk, L1ptr, 1, Mptr(A, ICOFF, jj + 1, lda), 1);
      }
      ii = iip1;
      iip1++;
      m = Mm1;
      Mm1--;
    }
    Nm1--;
    jj++;
  }
  /*
   * Swap and broadcast last row - Scale last column by its absolute value
   * max entry
   */
  if(thread_rank == 0) {
    HPL_pdmxswp(PANEL, m, ii, jj, WORK);
    HPL_dlocswpN(PANEL, ii, jj, WORK);
  }

#pragma omp barrier

  if(WORK[0] != HPL_rzero)
    HPL_dscal_omp(Mm1,
                  HPL_rone / WORK[0],
                  Mptr(A, iip1, jj, lda),
                  1,
                  PANEL->nb,
                  iip1,
                  thread_rank,
                  thread_size);

#ifdef HPL_DETAILED_TIMING
  if(thread_rank == 0) HPL_ptimer(HPL_TIMING_PFACT);
#endif
}
