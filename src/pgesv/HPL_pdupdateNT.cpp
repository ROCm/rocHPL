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

void HPL_pdupdateNT(HPL_T_panel* PANEL,
                    const int N,
                    double*   U,
                    const int LDU,
                    double*   A,
                    const int LDA,
                    const hipEvent_t& gemmStart,
                    const hipEvent_t& gemmStop) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdupdateNT applies the row interchanges and updates part of the
   * trailing  (using the panel PANEL) submatrix.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel (to be updated) information.
   *
   * ---------------------------------------------------------------------
   */

  /*
   * There is nothing to update, enforce the panel broadcast.
   */
  const int jb   = PANEL->jb;
  if((N <= 0) || (jb <= 0)) { return; }

  /* ..
   * .. Executable Statements ..
   */
  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  const double* L2 = PANEL->L2;
  const double* L1 = PANEL->L1;

  const int curr = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
  const int ldl2 = PANEL->ldl2;
  const int mp   = PANEL->mp - (curr != 0 ? jb : 0);

  const double one  = 1.0;
  const double mone = -1.0;

  /*
   * Update
   */
  if(PANEL->grid->nprow == 1) HPL_dlatcpy_gpu(N, jb, A, LDA, U, LDU);

  /*
   * Compute redundantly row block of U and update trailing submatrix
   */
  CHECK_ROCBLAS_ERROR(rocblas_dtrsm(handle,
                                    rocblas_side_right,
                                    rocblas_fill_lower,
                                    rocblas_operation_transpose,
                                    rocblas_diagonal_unit,
                                    N,
                                    jb,
                                    &one,
                                    L1,
                                    jb,
                                    U,
                                    LDU));

  /*
   * Queue finishing the update
   */
  if(curr != 0) {
    CHECK_HIP_ERROR(hipEventRecord(gemmStart, stream));
    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_transpose,
                                      mp,
                                      N,
                                      jb,
                                      &mone,
                                      L2,
                                      ldl2,
                                      U,
                                      LDU,
                                      &one,
                                      Mptr(A, jb, 0, LDA),
                                      LDA));
    CHECK_HIP_ERROR(hipEventRecord(gemmStop, stream));

    HPL_dlatcpy_gpu(jb, N, U, LDU, A, LDA);
  } else {
    CHECK_HIP_ERROR(hipEventRecord(gemmStart, stream));
    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_transpose,
                                      mp,
                                      N,
                                      jb,
                                      &mone,
                                      L2,
                                      ldl2,
                                      U,
                                      LDU,
                                      &one,
                                      A,
                                      LDA));
    CHECK_HIP_ERROR(hipEventRecord(gemmStop, stream));
  }
}
