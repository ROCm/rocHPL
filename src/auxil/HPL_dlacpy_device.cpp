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
#include <hip/hip_runtime.h>

#define TILE_DIM 64
#define BLOCK_ROWS 16

__global__ void dlacpy_gpu(const int M,
                           const int N,
                           const double* __restrict__ A,
                           const int LDA,
                           double* __restrict__ B,
                           const int LDB) {

  const int I = blockIdx.x * TILE_DIM + threadIdx.x;
  const int J = blockIdx.y * TILE_DIM + threadIdx.y;

  if(I < M) {
    if(J +  0 < N)
      B[I + static_cast<size_t>(LDB) * (J +  0)] = A[I + static_cast<size_t>(LDA) * (J +  0)];
    if(J + 16 < N)
      B[I + static_cast<size_t>(LDB) * (J + 16)] = A[I + static_cast<size_t>(LDA) * (J + 16)];
    if(J + 32 < N)
      B[I + static_cast<size_t>(LDB) * (J + 32)] = A[I + static_cast<size_t>(LDA) * (J + 32)];
    if(J + 48 < N)
      B[I + static_cast<size_t>(LDB) * (J + 48)] = A[I + static_cast<size_t>(LDA) * (J + 48)];
  }
}

void HPL_dlacpy_gpu(const int     M,
                    const int     N,
                    const double* A,
                    const int     LDA,
                    double*       B,
                    const int     LDB) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlacpy copies an array A into an array B.
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the number of rows of the arrays A and
   *         B. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies  the number of columns of the arrays A
   *         and B. N must be at least zero.
   *
   * A       (local input)                 const double *
   *         On entry, A points to an array of dimension (LDA,N).
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * B       (local output)                double *
   *         On entry, B points to an array of dimension (LDB,N). On exit,
   *         B is overwritten with A.
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
  dlacpy_gpu<<<grid_size, block_size, 0, stream>>>(M, N, A, LDA, B, LDB);
  CHECK_HIP_ERROR(hipGetLastError());
}
