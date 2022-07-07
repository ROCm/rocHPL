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

double HPL_dlange(const HPL_T_NORM NORM,
                  const int        M,
                  const int        N,
                  const double*    A,
                  const int        LDA) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlange returns  the value of the one norm,  or the infinity norm,
   * or the element of largest absolute value of a matrix A:
   *
   *    max(abs(A(i,j))) when NORM = HPL_NORM_A,
   *    norm1(A),        when NORM = HPL_NORM_1,
   *    normI(A),        when NORM = HPL_NORM_I,
   *
   * where norm1 denotes the one norm of a matrix (maximum column sum) and
   * normI denotes  the infinity norm of a matrix (maximum row sum).  Note
   * that max(abs(A(i,j))) is not a matrix norm.
   *
   * Arguments
   * =========
   *
   * NORM    (local input)                 const HPL_T_NORM
   *         On entry,  NORM  specifies  the  value to be returned by this
   *         function as described above.
   *
   * M       (local input)                 const int
   *         On entry,  M  specifies  the number  of rows of the matrix A.
   *         M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * A       (local input)                 const double *
   *         On entry,  A  points to an  array of dimension  (LDA,N), that
   *         contains the matrix A.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,M).
   *
   * ---------------------------------------------------------------------
   */

  double s, v0 = HPL_rzero, *work = NULL;
  int    i, j;

  if((M <= 0) || (N <= 0)) return (HPL_rzero);

  if(NORM == HPL_NORM_A) {
    /*
     * max( abs( A ) )
     */
    for(j = 0; j < N; j++) {
      for(i = 0; i < M; i++) {
        v0 = Mmax(v0, Mabs(*A));
        A++;
      }
      A += LDA - M;
    }
  } else if(NORM == HPL_NORM_1) {
    /*
     * Find norm_1( A ).
     */
    work = (double*)malloc((size_t)(N) * sizeof(double));
    if(work == NULL) {
      HPL_abort(__LINE__, "HPL_dlange", "Memory allocation failed");
    } else {
      for(j = 0; j < N; j++) {
        s = HPL_rzero;
        for(i = 0; i < M; i++) {
          s += Mabs(*A);
          A++;
        }
        work[j] = s;
        A += LDA - M;
      }
      /*
       * Find maximum sum of columns for 1-norm
       */
      v0 = work[HPL_idamax(N, work, 1)];
      v0 = Mabs(v0);
      if(work) free(work);
    }
  } else if(NORM == HPL_NORM_I) {
    /*
     * Find norm_inf( A )
     */
    work = (double*)malloc((size_t)(M) * sizeof(double));
    if(work == NULL) {
      HPL_abort(__LINE__, "HPL_dlange", "Memory allocation failed");
    } else {
      for(i = 0; i < M; i++) { work[i] = HPL_rzero; }

      for(j = 0; j < N; j++) {
        for(i = 0; i < M; i++) {
          work[i] += Mabs(*A);
          A++;
        }
        A += LDA - M;
      }
      /*
       * Find maximum sum of rows for inf-norm
       */
      v0 = work[HPL_idamax(M, work, 1)];
      v0 = Mabs(v0);
      if(work) free(work);
    }
  }

  return (v0);
}
