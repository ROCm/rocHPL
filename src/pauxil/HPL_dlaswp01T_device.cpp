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

/* Build U matrix from rows of A */
__global__ void dlaswp01T(const int M,
                          const int N,
                          double* __restrict__ A,
                          const int LDA,
                          double* __restrict__ U,
                          const int LDU,
                          const int* __restrict__ LINDXU) {

  __shared__ double s_U[TILE_DIM][TILE_DIM + 1];

  const int m = threadIdx.x + TILE_DIM * blockIdx.x;
  const int n = threadIdx.y + TILE_DIM * blockIdx.y;

  if(m < M) {
    const int ipa = LINDXU[m];

    // save in LDS for the moment
    // possible cache-hits if ipas are close
    s_U[threadIdx.x][threadIdx.y + 0] =
        (n + 0 < N) ? A[ipa + (n + 0) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 8] =
        (n + 8 < N) ? A[ipa + (n + 8) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 16] =
        (n + 16 < N) ? A[ipa + (n + 16) * ((size_t)LDA)] : 0.0;
    s_U[threadIdx.x][threadIdx.y + 24] =
        (n + 24 < N) ? A[ipa + (n + 24) * ((size_t)LDA)] : 0.0;
  }

  __syncthreads();

  const int um = threadIdx.y + TILE_DIM * blockIdx.x;
  const int un = threadIdx.x + TILE_DIM * blockIdx.y;

  if(un < N) {
    // write out chunks of U
    if((um + 0) < M)
      U[un + (um + 0) * ((size_t)LDU)] = s_U[threadIdx.y + 0][threadIdx.x];
    if((um + 8) < M)
      U[un + (um + 8) * ((size_t)LDU)] = s_U[threadIdx.y + 8][threadIdx.x];
    if((um + 16) < M)
      U[un + (um + 16) * ((size_t)LDU)] = s_U[threadIdx.y + 16][threadIdx.x];
    if((um + 24) < M)
      U[un + (um + 24) * ((size_t)LDU)] = s_U[threadIdx.y + 24][threadIdx.x];
  }
}

void HPL_dlaswp01T(const int  M,
                   const int  N,
                   double*    A,
                   const int  LDA,
                   double*    U,
                   const int  LDU,
                   const int* LINDXU) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaswp01T copies  scattered rows  of  A  into an array U.  The
   * row offsets in  A  of the source rows  are specified by LINDXU.
   * Rows of A are stored as columns in U.
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
   * U       (local input/output)          double *
   *         On entry, U points to an array of dimension (LDU,M). The rows
   *         of A specified by  LINDXA  are copied within this array  U at
   *         the  positions indicated by positive values of LINDXAU.  The
   *         rows of A are stored as columns in U.
   *
   * LDU     (local input)                 const int
   *         On entry, LDU specifies the leading dimension of the array U.
   *         LDU must be at least MAX(1,N).
   *
   * LINDXU  (local input)                 const int *
   *         On entry, LINDXU is an array of dimension M that contains the
   *         local  row indexes  of  A  that should be copied into U.
   *
   * ---------------------------------------------------------------------
   */
  /*
   * .. Local Variables ..
   */

  if((M <= 0) || (N <= 0)) return;

  dim3 grid_size((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
  dim3 block_size(TILE_DIM, BLOCK_ROWS);
  dlaswp01T<<<grid_size, block_size, 0, computeStream>>>(
      M, N, A, LDA, U, LDU, LINDXU);
  CHECK_HIP_ERROR(hipGetLastError());

  /*
   * End of HPL_dlaswp01T
   */
}
