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

#define BLOCK_SIZE 512

__global__ void dlaswp10N(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ IPIV) {

  const int m = threadIdx.x + BLOCK_SIZE * blockIdx.x;

  if(m < M) {
    for(int i = 0; i < N; i++) {
      const int ip = IPIV[i];

      if(ip != i) {
        // swap
        const double Ai           = A[m + i * ((size_t)LDA)];
        const double Aip          = A[m + ip * ((size_t)LDA)];
        A[m + i * ((size_t)LDA)]  = Aip;
        A[m + ip * ((size_t)LDA)] = Ai;
      }
    }
  }
}

void HPL_dlaswp10N(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* IPIV) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp10N performs a sequence  of  local column interchanges on a
   * matrix A.  One column interchange is initiated  for columns 0 through
   * N-1 of A.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         __arg0__
   *
   * N       (local input)                 const int
   *         On entry,  M  specifies  the number of rows of the array A. M
   *         must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, N specifies the number of columns of the array A. N
   *         must be at least zero.
   *
   * LDA     (local input)                 const int
   *         On entry, A  points to an  array of  dimension (LDA,N).  This
   *         array contains the columns onto which the interchanges should
   *         be applied. On exit, A contains the permuted matrix.
   *
   * IPIV    (local input)                 const int *
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  dim3 grid_size((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dlaswp10N<<<grid_size, dim3(BLOCK_SIZE), 0, stream>>>(M, N, A, LDA, IPIV);
  CHECK_HIP_ERROR(hipGetLastError());
}
