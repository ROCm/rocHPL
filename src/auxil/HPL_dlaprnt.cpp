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

void HPL_dlaprnt(const int   M,
                 const int   N,
                 double*     A,
                 const int   IA,
                 const int   JA,
                 const int   LDA,
                 const char* CMATNM) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlaprnt prints to standard error an M-by-N matrix A.
   *
   *
   * Arguments
   * =========
   *
   * M       (local input)                 const int
   *         On entry,  M  specifies the number of rows of A. M must be at
   *         least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies the number of columns of A. N must be
   *         at least zero.
   *
   * A       (local input)                 double *
   *         On entry, A  points to an array of dimension (LDA,N).
   *
   * IA      (local input)                 const int
   *         On entry, IA specifies the starting row index to be printed.
   *
   * JA      (local input)                 const int
   *         On entry,  JA  specifies  the  starting  column index  to be
   *         printed.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,M).
   *
   * CMATNM  (local input)                 const char *
   *         On entry, CMATNM is the name of the matrix to be printed.
   *
   * ---------------------------------------------------------------------
   */

  int i, j;

  for(j = 0; j < N; j++) {
    for(i = 0; i < M; i++) {
      HPL_fprintf(stderr,
                  "%s(%6d,%6d)=%30.18f\n",
                  CMATNM,
                  IA + i,
                  JA + j,
                  *(Mptr(A, i, j, LDA)));
    }
  }
}
