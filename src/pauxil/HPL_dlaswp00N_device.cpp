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

__global__ void dlaswp00N(const int N,
                          const int M,
                          double* __restrict__ A,
                          const int LDA,
                          const int* __restrict__ IPIV) {

  __shared__ double s_An_init[2048];
  __shared__ double s_An_ipiv[2048];

  const int m = threadIdx.x;
  const int n = blockIdx.x;

  // read in block column
  for(int i = m; i < M; i += blockDim.x)
    s_An_init[i] = A[i + n * ((size_t)LDA)];

  __syncthreads();

  // local block
  for(int i = m; i < M; i += blockDim.x) {
    const int ip = IPIV[i];

    if(ip < M) { // local swap
      s_An_ipiv[i] = s_An_init[ip];
    } else { // non local swap
      s_An_ipiv[i] = A[ip + n * ((size_t)LDA)];
    }
  }
  __syncthreads();

  // write out local block
  for(int i = m; i < M; i += blockDim.x)
    A[i + n * ((size_t)LDA)] = s_An_ipiv[i];

  // remaining swaps in column
  for(int i = m; i < M; i += blockDim.x) {
    const int ip_ex = IPIV[i + M];

    if(ip_ex > -1) { A[ip_ex + n * ((size_t)LDA)] = s_An_init[i]; }
  }
}

void HPL_dlaswp00N(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   const int* IPIV) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp00N performs a series of local row interchanges on a matrix
   * A. One row interchange is initiated for rows 0 through M-1 of A.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M specifies the number of rows of the array A to be
   *         interchanged. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N  specifies  the number of columns of the array A.
   *         N must be at least zero.
   *
   * A       (local input/output)          double *
   *         On entry, A  points to an array of dimension (LDA,N) to which
   *         the row interchanges will be  applied.  On exit, the permuted
   *         matrix.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * IPIV    (local input)                 const int *
   *         On entry,  IPIV  is  an  array of size  M  that  contains the
   *         pivoting  information.  For  k  in [0..M),  IPIV[k]=IROFF + l
   *         implies that local rows k and l are to be interchanged.
   *
   * ---------------------------------------------------------------------
   */

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  int grid_size = N;
  dlaswp00N<<<grid_size, BLOCK_SIZE, 0, stream>>>(N, M, A, LDA, IPIV);
  CHECK_HIP_ERROR(hipGetLastError());
}
