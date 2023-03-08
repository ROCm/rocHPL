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

void HPL_pdpanrlN(HPL_T_panel* PANEL,
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
   * HPL_pdpanrlN factorizes  a panel of columns  that is a sub-array of a
   * larger one-dimensional panel A using the Right-looking variant of the
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

  double *A, *Acur, *Anxt;
  int     Mm1, Nm1, curr, ii, iip1, jj, lda, m = M;
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  if(thread_rank == 0) HPL_ptimer(HPL_TIMING_PFACT);
#endif
  A    = PANEL->A;
  lda  = PANEL->lda;
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

  int nb = PANEL->nb;
  int jb = PANEL->jb;
  double* Amax = WORK + 2*(4+2*nb);

  /*
   * Find local absolute value max in first column - initialize WORK
   */
  HPL_dlocmax(
      PANEL, m, ii, jj, WORK, thread_rank, thread_size, max_index, max_value);

  /*
   * Swap and broadcast the current row
   */
  if(thread_rank == 0) {
    HPL_pdmxswp(PANEL, m, ii, jj, WORK);

    /*
     * Replicated copy of the current (new) row of A into L1
     */
    double *L = Mptr(PANEL->L1, jj, 0, jb);
    double *Wmx = WORK + 4;
    HPL_dcopy(jb, Wmx, 1, L, jb);

    /*
     * Local row swap in A
     */
    HPL_dlocswpN(PANEL, ii, jj, WORK);
  }

  #pragma omp barrier

  while(Nm1 >= 1) {
    Acur = Mptr(A, iip1, jj, lda);
    Anxt = Mptr(Acur, 0, 1, lda);

    /*
     * Scale current column by its absolute value max entry  -  Update trai-
     * ling sub-matrix and find local absolute value max in next column (On-
     * ly one pass through cache for each current column).  This sequence of
     * operations could benefit from a specialized blocked implementation.
     */
    if(WORK[0] != HPL_rzero)
      HPL_dscal_omp(Mm1,
                    HPL_rone / WORK[0],
                    Acur,
                    1,
                    PANEL->nb,
                    iip1,
                    thread_rank,
                    thread_size);

    HPL_daxpy_omp(Mm1,
                  -WORK[4 + jj + 1],
                  Acur,
                  1,
                  Anxt,
                  1,
                  PANEL->nb,
                  iip1,
                  thread_rank,
                  thread_size);

    if(thread_rank == 0) {
      // Duplicate the max row from the WORK space into Amax
      HPL_dcopy(PANEL->jb, WORK+4, 1, Amax, 1);
    }

    HPL_dlocmax(PANEL,
                Mm1,
                iip1,
                jj + 1,
                WORK,
                thread_rank,
                thread_size,
                max_index,
                max_value);

    // Wait for WORK to be populated, and Amax copied
    #pragma omp barrier

    /*
     * Use Amax to perform rank 1 update on other threads
     */
    if(Nm1 > 1)
      HPL_dger_omp(HplColumnMajor,
                   Mm1,
                   Nm1 - 1,
                   -HPL_rone,
                   Acur,
                   1,
                   Amax + jj + 2,
                   1,
                   Mptr(Anxt, 0, 1, lda),
                   lda,
                   PANEL->nb,
                   iip1,
                   thread_rank,
                   thread_size);

    /*
     * Thread 0 Swap and broadcast the current row
     */
    if(thread_rank == 0) {
      double* Wmx = WORK + 4;
      double* A0  = Wmx + nb;

      // Update the rows to swap
      HPL_daxpy(Nm1 - 1, -Wmx[jj], Amax + jj + 2, 1, Wmx + jj + 2, 1);
      if(curr) {
        HPL_daxpy(Nm1 - 1, -A0[jj], Amax + jj + 2, 1, A0 + jj + 2, 1);
      }

      // Swap
      HPL_pdmxswp(PANEL, m, iip1, jj+1, WORK);

      /*
       * Replicated copy of the current (new) row of A into L1
       */
      double *L = Mptr(PANEL->L1, jj + 1, 0, jb);
      HPL_dcopy(jb, Wmx, 1, L, jb);
    }

    // Wait for rank 1 update to complete
    #pragma omp barrier

    if(thread_rank == 0) {
      /*
       * Local row swap in A
       */
      HPL_dlocswpN(PANEL, iip1, jj+1, WORK);
    }

    if(curr != 0) {
      ii = iip1;
      iip1++;
      m = Mm1;
      Mm1--;
    }

    Nm1--;
    jj++;

    // Wait for local swap
    #pragma omp barrier
  }

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
