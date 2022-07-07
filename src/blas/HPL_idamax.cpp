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

void HPL_idamax_omp(const int     N,
                    const double* X,
                    const int     INCX,
                    const int     NB,
                    const int     II,
                    const int     thread_rank,
                    const int     thread_size,
                    int*          max_index,
                    double*       max_value) {

  max_index[thread_rank] = 0;
  max_value[thread_rank] = 0.0;

  if(N < 1) return;

  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int nn           = Mmin(NB - II, N);
    max_index[thread_rank] = HPL_idamax(nn, X, INCX);
    max_value[thread_rank] = X[max_index[thread_rank] * INCX];
  }
  ++tile;
  int i = NB - II;
  for(; i < N; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int nn  = Mmin(NB, N - i);
      const int idm = HPL_idamax(nn, X + i * INCX, INCX);
      if(abs(X[(idm + i) * INCX]) > abs(max_value[thread_rank])) {
        max_value[thread_rank] = X[(idm + i) * INCX];
        max_index[thread_rank] = idm + i;
      }
    }
    ++tile;
  }

#pragma omp barrier

  // finish reduction
  if(thread_rank == 0) {
    for(int rank = 1; rank < thread_size; ++rank) {
      if(abs(max_value[rank]) > abs(max_value[0])) {
        max_value[0] = max_value[rank];
        max_index[0] = max_index[rank];
      }
    }
  }
}
