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

void HPL_pdlaswp_start(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {
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
  double *A, *U, *W;
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork,
      *lindxU = NULL, *lindxA = NULL, *lindxAU, *permU, *permU_ex;
  int  icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW;

  /* ..
   * .. Executable Statements ..
   */
  n  = PANEL->n;
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  iflag = PANEL->IWORK;

  MPI_Comm comm = PANEL->grid->col_comm;

  // quick return if we're 1xQ
  if(nprow == 1) return;

  A      = PANEL->A;
  lda     = PANEL->lda;
  icurrow = PANEL->prow;

  if(UPD == HPL_LOOK_AHEAD) {
    U   = PANEL->U;
    W   = PANEL->W;
    LDU = PANEL->ldu0;
    LDW = PANEL->ldu0;
    n   = PANEL->nu0;

  } else if(UPD == HPL_UPD_1) {
    U   = PANEL->U1;
    W   = PANEL->W1;
    LDU = PANEL->ldu1;
    LDW = PANEL->ldu1;
    n   = PANEL->nu1;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0, lda);

  } else if(UPD == HPL_UPD_2) {
    U   = PANEL->U2;
    W   = PANEL->W2;
    LDU = PANEL->ldu2;
    LDW = PANEL->ldu2;
    n   = PANEL->nu2;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0 + PANEL->nu1, lda);
  }

  /*
   * Quick return if there is nothing to do
   */
  if((n <= 0) || (jb <= 0)) return;

  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k         = (int)((unsigned int)(jb) << 1);
  ipl       = iflag + 1;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  lindxU  = PANEL->lindxU;
  lindxA  = PANEL->lindxA;
  lindxAU = PANEL->lindxAU;
  permU   = PANEL->permU;
  permU_ex = permU + jb;

  if(*iflag == -1) /* no index arrays have been computed so far */
  {
    // compute spreading info
    HPL_pipid(PANEL, ipl, ipID);
    HPL_plindx(
        PANEL, *ipl, ipID, ipA, lindxU, lindxAU, lindxA, iplen, permU, iwork);
    *iflag = 1;
  }

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */
  if(myrow == icurrow) {
    // copy needed rows of A into U
    HPL_dlaswp01T(jb, n, A, lda, U, LDU, lindxU);
  } else {
    // copy needed rows from A into U(:, iplen[myrow])
    HPL_dlaswp03T(iplen[myrow + 1] - iplen[myrow],
                  n,
                  A,
                  lda,
                  Mptr(U, 0, iplen[myrow], LDU),
                  LDU,
                  lindxU);
  }

  // record when packing completes
  CHECK_HIP_ERROR(hipEventRecord(swapStartEvent[UPD], computeStream));

  /*
   * End of HPL_pdlaswp_start
   */
}

void HPL_pdlaswp_exchange(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {
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
  double *A, *U, *W;
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork;
  int *lindxU = NULL, *lindxA = NULL, *lindxAU, *permU, *permU_ex;
  int  icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW;

  /* ..
   * .. Executable Statements ..
   */
  n  = PANEL->n;
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  iflag = PANEL->IWORK;

  MPI_Comm comm = PANEL->grid->col_comm;

  // quick return if we're 1xQ
  if(nprow == 1) return;

  A      = PANEL->A;
  lda     = PANEL->lda;
  icurrow = PANEL->prow;

  if(UPD == HPL_LOOK_AHEAD) {
    U   = PANEL->U;
    W   = PANEL->W;
    LDU = PANEL->ldu0;
    LDW = PANEL->ldu0;
    n   = PANEL->nu0;

  } else if(UPD == HPL_UPD_1) {
    U   = PANEL->U1;
    W   = PANEL->W1;
    LDU = PANEL->ldu1;
    LDW = PANEL->ldu1;
    n   = PANEL->nu1;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0, lda);

  } else if(UPD == HPL_UPD_2) {
    U   = PANEL->U2;
    W   = PANEL->W2;
    LDU = PANEL->ldu2;
    LDW = PANEL->ldu2;
    n   = PANEL->nu2;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0 + PANEL->nu1, lda);
  }

  /*
   * Quick return if there is nothing to do
   */
  if((n <= 0) || (jb <= 0)) return;

  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k         = (int)((unsigned int)(jb) << 1);
  ipl       = iflag + 1;
  ipID      = ipl + 1;
  ipA       = ipID + ((unsigned int)(k) << 1);
  iplen     = ipA + 1;
  ipcounts  = iplen + nprow + 1;
  ipoffsets = ipcounts + nprow;
  iwork     = ipoffsets + nprow;

  lindxA  = PANEL->lindxA;
  lindxAU = PANEL->lindxAU;
  lindxU  = PANEL->lindxU;
  permU   = PANEL->permU;
  permU_ex = permU + jb;

  /* Set MPI message counts and offsets */
  ipcounts[0]  = (iplen[1] - iplen[0]) * LDU;
  ipoffsets[0] = 0;

  for(int i = 1; i < nprow; ++i) {
    ipcounts[i]  = (iplen[i + 1] - iplen[i]) * LDU;
    ipoffsets[i] = ipcounts[i - 1] + ipoffsets[i - 1];
  }
  ipoffsets[nprow] = ipcounts[nprow - 1] + ipoffsets[nprow - 1];

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */

  if(myrow == icurrow) {

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
#endif

    // hipStreamSynchronize(computeStream);
    CHECK_HIP_ERROR(hipEventSynchronize(swapStartEvent[UPD]));

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
    HPL_ptimer(HPL_TIMING_LASWP);
#endif

    // send rows to other ranks
    HPL_scatterv(U, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);

    // All gather U
    HPL_allgatherv(U, ipcounts[myrow], ipcounts, ipoffsets, comm);

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_LASWP);
#endif

  } else {

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
#endif

    // wait for U to be ready
    // hipStreamSynchronize(computeStream);
    CHECK_HIP_ERROR(hipEventSynchronize(swapStartEvent[UPD]));

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
    HPL_ptimer(HPL_TIMING_LASWP);
#endif

    // receive rows from icurrow into W
    HPL_scatterv(W, ipcounts, ipoffsets, ipcounts[myrow], icurrow, comm);

    // All gather U
    HPL_allgatherv(U, ipcounts[myrow], ipcounts, ipoffsets, comm);

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_LASWP);
#endif
  }
  /*
   * End of HPL_pdlaswp_exchange
   */
}

void HPL_pdlaswp_end(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {
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
  double *A, *U, *W;
  int *   ipID, *iplen, *ipcounts, *ipoffsets, *iwork;
  int *lindxA = NULL, *lindxAU, *lindxU, *permU, *permU_ex;
  int  icurrow, *iflag, *ipA, *ipl, jb, k, lda, myrow, n, nprow, LDU, LDW;

  /* ..
   * .. Executable Statements ..
   */
  n  = PANEL->n;
  jb = PANEL->jb;

  /*
   * Retrieve parameters from the PANEL data structure
   */
  nprow = PANEL->grid->nprow;
  myrow = PANEL->grid->myrow;
  iflag = PANEL->IWORK;

  MPI_Comm comm = PANEL->grid->col_comm;

  A      = PANEL->A;
  lda     = PANEL->lda;
  icurrow = PANEL->prow;

  if(UPD == HPL_LOOK_AHEAD) {
    U   = PANEL->U;
    W   = PANEL->W;
    LDU = PANEL->ldu0;
    LDW = PANEL->ldu0;
    n   = PANEL->nu0;

  } else if(UPD == HPL_UPD_1) {
    U   = PANEL->U1;
    W   = PANEL->W1;
    LDU = PANEL->ldu1;
    LDW = PANEL->ldu1;
    n   = PANEL->nu1;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0, lda);

  } else if(UPD == HPL_UPD_2) {
    U   = PANEL->U2;
    W   = PANEL->W2;
    LDU = PANEL->ldu2;
    LDW = PANEL->ldu2;
    n   = PANEL->nu2;
    // we call the row swap start before the first section is updated
    //  so shift the pointers
    A = Mptr(A, 0, PANEL->nu0 + PANEL->nu1, lda);
  }

  /*
   * Quick return if there is nothing to do
   */
  if((n <= 0) || (jb <= 0)) return;

  // just local swaps if we're 1xQ
  if(nprow == 1) {
    HPL_dlaswp00N(jb, n, A, lda, PANEL->ipiv);
    return;
  }

  /*
   * Compute ipID (if not already done for this panel). lindxA and lindxAU
   * are of length at most 2*jb - iplen is of size nprow+1, ipmap, ipmapm1
   * are of size nprow,  permU is of length jb, and  this function needs a
   * workspace of size max( 2 * jb (plindx1), nprow+1(equil)):
   * 1(iflag) + 1(ipl) + 1(ipA) + 9*jb + 3*nprow + 1 + MAX(2*jb,nprow+1)
   * i.e. 4 + 9*jb + 3*nprow + max(2*jb, nprow+1);
   */
  k     = (int)((unsigned int)(jb) << 1);
  ipl   = iflag + 1;
  ipID  = ipl + 1;
  ipA   = ipID + ((unsigned int)(k) << 1);
  iplen = ipA + 1;

  lindxA  = PANEL->lindxA;
  lindxAU = PANEL->lindxAU;
  permU   = PANEL->permU;

  lindxA   = PANEL->lindxA;
  lindxAU  = PANEL->lindxAU;
  lindxU   = PANEL->lindxU;
  permU    = PANEL->permU;
  permU_ex = permU + jb;

  /*
   * For i in [0..2*jb),  lindxA[i] is the offset in A of a row that ulti-
   * mately goes to U( :, lindxAU[i] ).  In each rank, we directly pack
   * into U, otherwise we pack into workspace. The  first
   * entry of each column packed in workspace is in fact the row or column
   * offset in U where it should go to.
   */

  if(myrow == icurrow) {
    // swap rows local to A on device
    HPL_dlaswp02T(*ipA, n, A, lda, lindxAU, lindxA);
  } else {
    // Queue inserting recieved rows in W into A on device
    HPL_dlaswp04T(
        iplen[myrow + 1] - iplen[myrow], n, A, lda, W, LDW, lindxU);
  }

  /*
   * Permute U in every process row
   */
  HPL_dlaswp10N(n, jb, U, LDU, permU);
  /*
   * End of HPL_pdlaswp_endT
   */
}
