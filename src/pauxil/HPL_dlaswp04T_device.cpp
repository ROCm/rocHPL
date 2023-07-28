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

#define TILE_DIM 32
#define BLOCK_ROWS 8

static __global__ void dlaswp04T(const int M,
                                 const int N,
                                 double* __restrict__ A,
                                 const int LDA,
                                 double* __restrict__ W,
                                 const int LDW,
                                 const int* __restrict__ LINDXU) {

  __shared__ double s_W[TILE_DIM][TILE_DIM + 1];

  const int am = threadIdx.x + TILE_DIM * blockIdx.x;
  const int an = threadIdx.y + TILE_DIM * blockIdx.y;

  const int wm = threadIdx.y + TILE_DIM * blockIdx.x;
  const int wn = threadIdx.x + TILE_DIM * blockIdx.y;

  if(wn < N) {
    s_W[threadIdx.y + 0][threadIdx.x] =
        (wm + 0 < M) ? W[wn + (wm + 0) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 8][threadIdx.x] =
        (wm + 8 < M) ? W[wn + (wm + 8) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 16][threadIdx.x] =
        (wm + 16 < M) ? W[wn + (wm + 16) * ((size_t)LDW)] : 0.0;
    s_W[threadIdx.y + 24][threadIdx.x] =
        (wm + 24 < M) ? W[wn + (wm + 24) * ((size_t)LDW)] : 0.0;
  }

  __syncthreads();

  if(am < M) {
    const int aip = LINDXU[am];
    if((an + 0) < N)
      A[aip + (an + 0) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 0];
    if((an + 8) < N)
      A[aip + (an + 8) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 8];
    if((an + 16) < N)
      A[aip + (an + 16) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 16];
    if((an + 24) < N)
      A[aip + (an + 24) * ((size_t)LDA)] = s_W[threadIdx.x][threadIdx.y + 24];
  }
}

void HPL_dlaswp04T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    W,
                   const int  LDW,
                   const int* LINDXU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp04T writes columns  of  W  into  rows  of  A  at  positions
   * indicated by LINDXU.
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry, M  specifies the number of rows of A that should be
   *         replaced with columns of W. M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry, N specifies the length of the rows of A that should
   *         be replaced with columns of W. N must be at least zero.
   *
   * A       (local output)                double *
   *         On entry, A points to an array of dimension (LDA,N). On exit,
   *         the  rows of this array specified by  LINDXU  are replaced by
   *         columns of W.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least MAX(1,M).
   *
   * W       (local input/output)          double *
   *         On entry,  W  points  to an array of dimension (LDW,*).  This
   *         array contains the columns of  W  that are to be writen to
   *         rows of A.
   *
   * LDW     (local input)                 const int
   *         On entry, LDW specifies the leading dimension of the array W.
   *         LDW must be at least MAX(1,N).
   *
   * LINDXU  (local input)                 const int *
   *         On entry, LINDXU is an array of dimension M that contains the
   *         local row indexes of A that should be replaced with W.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
  dim3 block_size(TILE_DIM, BLOCK_ROWS);
  dlaswp04T<<<grid_size, block_size, 0, computeStream>>>(
      M, N, A, LDA, W, LDW, LINDXU);
  CHECK_HIP_ERROR(hipGetLastError());
  /*
   * End of HPL_dlaswp04T
   */
}
