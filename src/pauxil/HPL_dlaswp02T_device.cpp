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
#include <hip/hip_runtime.h>
#include <cassert>

#define assertm(exp, msg) assert(((void)msg, exp))

/* Perform any local row swaps of A */
__global__ void dlaswp02T(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ LINDXAU,
                          const int* __restrict__ LINDXA) {

  const int n = blockIdx.x;
  const int m = threadIdx.x;

  const int ipau = LINDXAU[m]; // src row
  const int ipa  = LINDXA[m];  // dst row

  const double An = A[ipau + n * ((size_t)LDA)];

  __syncthreads();

  A[ipa + n * ((size_t)LDA)] = An;
}

void HPL_dlaswp02T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* LINDXAU,
                   const int* LINDXA) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp02T copies  scattered rows  of  A  into itself. The row
   * offsets in  A  of the source rows  are specified by LINDXA.
   * The  destination of those rows are specified by  LINDXAU.  A
   * positive value of LINDXAU indicates that the array  destination is U,
   * and A otherwise.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of A that should be
   *         moved within A or copied into U. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N  specifies the length of rows of A that should be
   *         moved within A or copied into U. N must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, A points to an array of dimension (LDA,N). The rows
   *         of this array specified by LINDXA should be moved within A or
   *         copied into U.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * LINDXAU (local input)                 const int *
   *         On entry, LINDXA is an array of dimension M that contains the
   *         local  row indexes  of  A  that should be moved within  A.
   *
   * LINDXA  (local input)                 const int *
   *         On entry, LINDXAU  is an array of dimension  M that  contains
   *         the local  row indexes of  A  where the rows of  A  should be
   *         copied to.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  assertm(M <= 1024, "NB too large in HPL_dlaswp02T");

  dim3 grid_size(N);
  dim3 block_size(M);
  dlaswp02T<<<N, M, 0, computeStream>>>(M, N, A, LDA, LINDXAU, LINDXA);
  CHECK_HIP_ERROR(hipGetLastError());
  /*
   * End of HPL_dlaswp02T
   */
}
