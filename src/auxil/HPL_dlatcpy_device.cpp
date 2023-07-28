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

#define TILE_DIM 64
#define BLOCK_ROWS 16

__global__ void dlatcpy_gpu(const int M,
                            const int N,
                            const double* __restrict__ A,
                            const int LDA,
                            double* __restrict__ B,
                            const int LDB) {

  __shared__ double s_tile[TILE_DIM][TILE_DIM + 1];

  int I = blockIdx.x * TILE_DIM + threadIdx.y;
  int J = blockIdx.y * TILE_DIM + threadIdx.x;

  if(J < N) {
    if(I + 0 < M)
      s_tile[threadIdx.y + 0][threadIdx.x] = A[((size_t)I + 0) * LDA + J];
    if(I + 16 < M)
      s_tile[threadIdx.y + 16][threadIdx.x] = A[((size_t)I + 16) * LDA + J];
    if(I + 32 < M)
      s_tile[threadIdx.y + 32][threadIdx.x] = A[((size_t)I + 32) * LDA + J];
    if(I + 48 < M)
      s_tile[threadIdx.y + 48][threadIdx.x] = A[((size_t)I + 48) * LDA + J];
  }

  I = blockIdx.x * TILE_DIM + threadIdx.x;
  J = blockIdx.y * TILE_DIM + threadIdx.y;

  __syncthreads();

  if(I < M) {
    if(J + 0 < N)
      B[I + ((size_t)J + 0) * LDB] = s_tile[threadIdx.x][threadIdx.y + 0];
    if(J + 16 < N)
      B[I + ((size_t)J + 16) * LDB] = s_tile[threadIdx.x][threadIdx.y + 16];
    if(J + 32 < N)
      B[I + ((size_t)J + 32) * LDB] = s_tile[threadIdx.x][threadIdx.y + 32];
    if(J + 48 < N)
      B[I + ((size_t)J + 48) * LDB] = s_tile[threadIdx.x][threadIdx.y + 48];
  }
}

void HPL_dlatcpy_gpu(const int     M,
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

  if((M <= 0) || (N <= 0)) return;

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  dim3 grid_size((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
  dim3 block_size(TILE_DIM, BLOCK_ROWS);
  dlatcpy_gpu<<<grid_size, block_size, 0, stream>>>(M, N, A, LDA, B, LDB);
  CHECK_HIP_ERROR(hipGetLastError());
}
