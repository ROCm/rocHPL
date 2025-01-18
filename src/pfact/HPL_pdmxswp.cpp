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

void HPL_pdmxswp(HPL_T_panel* PANEL,
                 const int    M,
                 const int    II,
                 const int    JJ,
                 double*      WORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdmxswp swaps  and  broadcasts  the  absolute value max row using
   * bi-directional exchange.  The buffer is partially set by HPL_dlocmax.
   *
   * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
   * operations  at once  for one column in the panel.  This  results in a
   * lower number of slightly larger  messages than usual.  On P processes
   * and assuming bi-directional links,  the running time of this function
   * can be approximated by
   *
   *    log_2( P ) * ( lat + ( 2 * N0 + 4 ) / bdwth )
   *
   * where  lat and bdwth are the latency and bandwidth of the network for
   * double precision real elements.  Communication  only  occurs  in  one
   * process  column. Mono-directional links  will cause the communication
   * cost to double.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the local number of rows of the matrix
   *         column on which this function operates.
   *
   * II      (local input)                 const int
   *         On entry, II  specifies the row offset where the column to be
   *         operated on starts with respect to the panel.
   *
   * JJ      (local input)                 const int
   *         On entry, JJ  specifies the column offset where the column to
   *         be operated on starts with respect to the panel.
   *
   * WORK    (local workspace)             double *
   *         On entry, WORK  is a workarray of size at least 2 * (4+2*N0).
   *         It  is assumed that  HPL_dlocmax  was called  prior  to  this
   *         routine to  initialize  the first four entries of this array.
   *         On exit, the  N0  length max row is stored in WORK[4:4+N0-1];
   *         Note that this is also the  JJth  row  (or column) of L1. The
   *         remaining part is used as a temporary array.
   *
   * ---------------------------------------------------------------------
   */

  double *    A0, *Wmx, *Wwork;
  HPL_T_grid* grid;
  MPI_Comm    comm;
  int         cnt_, cnt0, i, icurrow, lda, myrow, n0;

/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
  grid    = PANEL->grid;
  comm    = grid->col_comm;
  myrow   = grid->myrow;
  n0      = PANEL->jb;
  int NB  = PANEL->nb;
  icurrow = PANEL->prow;
  /*
   * Set up pointers in workspace:  WORK and Wwork  point to the beginning
   * of the buffers of size 4 + 2*N0 to be combined. Wmx points to the row
   * owning the local (before combine) and global (after combine) absolute
   * value max. A0 points to the copy of the current row of the matrix.
   */
  cnt0 = 4 + 2 * NB;

  A0    = (Wmx = WORK + 4) + NB;
  Wwork = WORK + cnt0;

  /*
   * Wmx[0:N0-1] := A[ilindx,0:N0-1] where ilindx is  (int)(WORK[1])  (row
   * with max in current column). If I am the current process row, pack in
   * addition the current row of A in A0[0:N0-1].  If I do not own any row
   * of A, then zero out Wmx[0:N0-1].
   */
  if(M > 0) {
    lda = PANEL->lda0;

    HPL_dcopy(n0, Mptr(PANEL->hA0, II + (int)(WORK[1]), 0, lda), lda, Wmx, 1);
    if(myrow == icurrow) {
      HPL_dcopy(n0, Mptr(PANEL->hA0, II, 0, lda), lda, A0, 1);
    } else {
      for(i = 0; i < n0; i++) A0[i] = HPL_rzero;
    }
  } else {
    for(i = 0; i < n0; i++) A0[i] = HPL_rzero;
    for(i = 0; i < n0; i++) Wmx[i] = HPL_rzero;
  }

  /* Perform swap-broadcast */
  HPL_all_reduce_dmxswp(WORK, cnt0, icurrow, comm, Wwork);

  /*
   * Save the global pivot index in pivot array
   */
  (PANEL->ipiv)[JJ] = (int)WORK[2];
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
}
