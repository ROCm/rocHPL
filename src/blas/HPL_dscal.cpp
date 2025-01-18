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

void HPL_dscal_omp(const int    N,
                   const double ALPHA,
                   double*      X,
                   const int    INCX,
                   const int    NB,
                   const int    II,
                   const int    thread_rank,
                   const int    thread_size) {

  int tile = 0;
  if(tile % thread_size == thread_rank) {
    const int nn = Mmin(NB - II, N);
    HPL_dscal(nn, ALPHA, X, INCX);
  }
  ++tile;
  int i = NB - II;
  for(; i < N; i += NB) {
    if(tile % thread_size == thread_rank) {
      const int nn = Mmin(NB, N - i);
      HPL_dscal(nn, ALPHA, X + i * INCX, INCX);
    }
    ++tile;
  }
}
