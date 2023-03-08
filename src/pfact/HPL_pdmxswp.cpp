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

/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif

  HPL_T_grid *grid = PANEL->grid;
  MPI_Comm    comm = grid->col_comm;
  int NB      = PANEL->nb;
  int icurrow = PANEL->prow;

  int cnt0 = 4 + 2 * NB;
  double* Wwork = WORK + cnt0;

  /* Perform swap-broadcast */
  HPL_all_reduce_dmxswp(WORK, cnt0, icurrow, comm, Wwork);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
  /*
   * Save the global pivot index in pivot array
   */
  (PANEL->ipiv)[JJ] = (int)WORK[2];
}
