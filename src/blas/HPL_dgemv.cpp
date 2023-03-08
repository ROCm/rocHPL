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

void HPL_dgemv_omp(const enum HPL_ORDER ORDER,
                   const enum HPL_TRANS TRANS,
                   const int            M,
                   const int            N,
                   const double         ALPHA,
                   const double*        A,
                   const int            LDA,
                   const double*        X,
                   const int            INCX,
                   const double         BETA,
                   double*              Y,
                   const int            INCY,
                   const int            NB,
                   const int            II,
                   const int            thread_rank,
                   const int            thread_size) {

  if (thread_size==1) {

    HPL_dgemv(ORDER, TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);

  } else {

    if (thread_rank==0) return;

    int tile = 0;
    if(tile % (thread_size-1) == (thread_rank-1)) {
      const int mm = Mmin(NB - II, M);
      HPL_dgemv(ORDER, TRANS, mm, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
    }
    ++tile;
    int i = NB - II;
    for(; i < M; i += NB) {
      if(tile % (thread_size-1) == (thread_rank-1)) {
        const int mm = Mmin(NB, M - i);
        HPL_dgemv(ORDER,
                  TRANS,
                  mm,
                  N,
                  ALPHA,
                  A + i,
                  LDA,
                  X,
                  INCX,
                  BETA,
                  Y + i * INCY,
                  INCY);
      }
      ++tile;
    }
  }
}
