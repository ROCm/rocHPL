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

void HPL_pdlaswp_start(HPL_T_panel* PANEL,
                       const int N,
                       double*   U,
                       const int LDU,
                       double*   W,
                       const int LDW,
                       double*   A,
                       const int LDA,
                       const hipEvent_t& swapEvent) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswp_start begins the  NB  row interchanges to  NN columns of the
   * trailing submatrix and broadcast a column panel. The rows needed for
   * the row interchanges are packed into U (in the current row) and W
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork,
      *lindxU = NULL, *lindxA = NULL, *lindxAU, *permU;
  int icurrow, *iflag, *ipA, *ipl, jb, k, myrow, nprow;

  /* ..
   * .. Executable Statements ..
   */
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  MPI_Comm comm = PANEL->grid->col_comm;

  // quick return if we're 1xQ
  if(nprow == 1) return;

  /*
   * Quick return if there is nothing to do
   */
  if((N <= 0) || (jb <= 0)) return;

  icurrow  = PANEL->prow;
  permU    = PANEL->IWORK;
  lindxU   = permU + jb;
  lindxA   = lindxU + jb;
  lindxAU  = lindxA + jb;

  k         = (int)((unsigned int)(jb) << 1);
  ipl       = lindxAU + jb;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */
  if(myrow == icurrow) {
    // copy needed rows of A into U
    HPL_dlaswp01T(jb, N, A, LDA, U, LDU, lindxU);
  } else {
    // copy needed rows from A into U(:, iplen[myrow])
    HPL_dlaswp03T(iplen[myrow + 1] - iplen[myrow],
                  N,
                  A,
                  LDA,
                  Mptr(U, 0, iplen[myrow], LDU),
                  LDU,
                  lindxU);
  }

  // record when packing completes
  CHECK_HIP_ERROR(hipEventRecord(swapEvent, computeStream));

  /*
   * End of HPL_pdlaswp_start
   */
}

void HPL_pdlaswp_exchange(HPL_T_panel* PANEL,
                          const int N,
                          double*   U,
                          const int LDU,
                          double*   W,
                          const int LDW,
                          double*   A,
                          const int LDA,
                          const hipEvent_t& swapEvent) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswp_exchange applies the  NB  row interchanges to  NN columns of
   * the trailing submatrix and broadcast a column panel.
   *
   * A "Spread then roll" algorithm performs  the swap :: broadcast  of the
   * row panel U at once,  resulting in a minimal communication volume  and
   * a "very good"  use of the connectivity if available.  With  P  process
   * rows  and  assuming  bi-directional links,  the  running time  of this
   * function can be approximated by:
   *
   *    (log_2(P)+(P-1)) * lat +   K * NB * LocQ(N) / bdwth
   *
   * where  NB  is the number of rows of the row panel U,  N is the global
   * number of columns being updated,  lat and bdwth  are the latency  and
   * bandwidth  of  the  network  for  double  precision real words.  K is
   * a constant in (2,3] that depends on the achieved bandwidth  during  a
   * simultaneous  message exchange  between two processes.  An  empirical
   * optimistic value of K is typically 2.4.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork;
  int *   lindxU = NULL, *lindxA = NULL, *lindxAU, *permU;
  int     icurrow, *iflag, *ipA, *ipl, jb, k, myrow, nprow;

  /* ..
   * .. Executable Statements ..
   */
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  MPI_Comm comm = PANEL->grid->col_comm;

  // quick return if we're 1xQ
  if(nprow == 1) return;

  /*
   * Quick return if there is nothing to do
   */
  if((N <= 0) || (jb <= 0)) return;

  icurrow  = PANEL->prow;
  permU    = PANEL->IWORK;
  lindxU   = permU + jb;
  lindxA   = lindxU + jb;
  lindxAU  = lindxA + jb;

  k         = (int)((unsigned int)(jb) << 1);
  ipl       = lindxAU + jb;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  /* Set MPI message counts and offsets */
  ipcounts[0]  = (iplen[1] - iplen[0]) * LDU;
  ipoffsets[0] = 0;

  for(int i = 1; i < nprow; ++i) {
    ipcounts[i]  = (iplen[i + 1] - iplen[i]) * LDU;
    ipoffsets[i] = ipcounts[i - 1] + ipoffsets[i - 1];
  }
  ipoffsets[nprow] = ipcounts[nprow - 1] + ipoffsets[nprow - 1];


#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
  CHECK_HIP_ERROR(hipEventSynchronize(swapEvent));
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
  HPL_ptimer(HPL_TIMING_LASWP);
#endif

  if(myrow == icurrow) {
    // send rows to other ranks
    HPL_scatterv(U, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);

    // All gather U
    HPL_allgatherv(U, ipcounts[myrow], ipcounts, ipoffsets, comm);

  } else {
    // receive rows from icurrow into W
    HPL_scatterv(W, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);

    // All gather U
    HPL_allgatherv(U, ipcounts[myrow], ipcounts, ipoffsets, comm);
  }

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_LASWP);
#endif
  /*
   * End of HPL_pdlaswp_exchange
   */
}

void HPL_pdlaswp_end(HPL_T_panel* PANEL,
                     const int N,
                     double*   U,
                     const int LDU,
                     double*   W,
                     const int LDW,
                     double*   A,
                     const int LDA) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlaswp_end copies  scattered rows  of  A  into an array U.  The
   * row offsets in  A  of the source rows  are specified by LINDXA.  The
   * destination of those rows are specified by  LINDXAU.  A
   * positive value of LINDXAU indicates that the array  destination is U,
   * and A otherwise. Rows of A are stored as columns in U.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork;
  int *   lindxA = NULL, *lindxAU, *lindxU, *permU;
  int     icurrow, *iflag, *ipA, *ipl, jb, k, myrow, nprow;

  /* ..
   * .. Executable Statements ..
   */
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  MPI_Comm comm = PANEL->grid->col_comm;

  /*
   * Quick return if there is nothing to do
   */
  if((N <= 0) || (jb <= 0)) return;

  icurrow  = PANEL->prow;
  permU    = PANEL->IWORK;
  lindxU   = permU + jb;
  lindxA   = lindxU + jb;
  lindxAU  = lindxA + jb;

  k         = (int)((unsigned int)(jb) << 1);
  ipl       = lindxAU + jb;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  // just local swaps if we're 1xQ
  if(nprow == 1) {
    HPL_dlaswp00N(jb, N, A, LDA, permU);
    HPL_dlatcpy_gpu(N, jb, A, LDA, U, LDU);
    return;
  }

  if(myrow == icurrow) {
    // swap rows local to A on device
    HPL_dlaswp02T(*ipA, N, A, LDA, lindxAU, lindxA);
  } else {
    // Queue inserting recieved rows in W into A on device
    HPL_dlaswp04T(iplen[myrow + 1] - iplen[myrow], N, A, LDA, W, LDW, lindxU);
  }

  /*
   * Permute U in every process row
   */
  HPL_dlaswp10N(N, jb, U, LDU, permU);
  /*
   * End of HPL_pdlaswp_endT
   */
}
