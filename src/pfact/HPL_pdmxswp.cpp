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

  double *Wwork;
  HPL_T_grid* grid;
  MPI_Comm    comm;
  int         cnt0, icurrow, myrow, nprow;

/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_MXSWP);
#endif
  grid    = PANEL->grid;
  comm    = grid->col_comm;
  myrow   = grid->myrow;
  nprow   = grid->nprow;
  int NB  = PANEL->jb;
  icurrow = PANEL->prow;

  cnt0 = 4 + 2 * NB;
  Wwork = WORK + cnt0;

  if (M>0) {
    int ilindx = static_cast<int>(WORK[1]);
    int kk     = PANEL->ii + II + (ilindx);
    int igindx = 0;
    Mindxl2g(igindx, kk, NB, NB, myrow, 0, nprow);
    /*
     * WORK[0] := local maximum absolute value scalar,
     * WORK[1] := corresponding local  row index,
     * WORK[2] := corresponding global row index,
     * WORK[3] := coordinate of process owning this max.
     */
    WORK[2] = (double)(igindx);
    WORK[3] = (double)(myrow);

  } else {
    WORK[0] = WORK[1] = WORK[2] = HPL_rzero;
    WORK[3]                     = (double)(PANEL->grid->nprow);
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
