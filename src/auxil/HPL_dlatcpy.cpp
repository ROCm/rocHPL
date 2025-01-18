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

void HPL_dlatcpy(const int     M,
                 const int     N,
                 const double* A,
                 const int     LDA,
                 double*       B,
                 const int     LDB) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlatcpy copies the transpose of an array A into an array B.
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the number of  rows of the array B and
   *         the number of columns of A. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the number of  rows of the array A and
   *         the number of columns of B. N must be at least zero.
   *
   * A       (local input)                 const double *
   *         On entry, A points to an array of dimension (LDA,M).
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,N).
   *
   * B       (local output)                double *
   *         On entry, B points to an array of dimension (LDB,N). On exit,
   *         B is overwritten with the transpose of A.
   *
   * LDB     (local input)                 const int
   *         On entry, LDB specifies the leading dimension of the array B.
   *         LDB must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */
  int j;

  if((M <= 0) || (N <= 0)) return;

  for(j = 0; j < N; j++, B += LDB) HPL_dcopy(M, A + j, LDA, B, 1);
}
