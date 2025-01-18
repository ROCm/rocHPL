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

void HPL_perm(const int N, int* LINDXA, int* LINDXAU, int* IWORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_perm combines  two  index  arrays  and generate the corresponding
   * permutation. First, this function computes the inverse of LINDXA, and
   * then combine it with LINDXAU.  Second, in order to be able to perform
   * the permutation in place,  LINDXAU  is overwritten by the sequence of
   * permutation  producing  the  same result.  What we ultimately want to
   * achieve is:  U[LINDXAU[i]] := U[LINDXA[i]] for i in [0..N). After the
   * call to this function,  this in place permutation can be performed by
   * for i in [0..N) swap U[i] with U[LINDXAU[i]].
   *
   * Arguments
   * =========
   *
   * N       (global input)                const int
   *         On entry,  N  specifies the length of the arrays  LINDXA  and
   *         LINDXAU. N should be at least zero.
   *
   * LINDXA  (global input/output)         int *
   *         On entry,  LINDXA  is an array of dimension N  containing the
   *         source indexes. On exit,  LINDXA  contains the combined index
   *         array.
   *
   * LINDXAU (global input/output)         int *
   *         On entry,  LINDXAU is an array of dimension N  containing the
   *         target indexes.  On exit,  LINDXAU  contains  the sequence of
   *         permutation,  that  should be applied  in increasing order to
   *         permute the underlying array U in place.
   *
   * IWORK   (workspace)                   int *
   *         On entry, IWORK is a workarray of dimension N.
   *
   * ---------------------------------------------------------------------
   */

  int i, j, k, fndd;

  /*
   * Inverse LINDXA - combine LINDXA and LINDXAU - Initialize IWORK
   */
  for(i = 0; i < N; i++) { IWORK[LINDXA[i]] = i; }
  for(i = 0; i < N; i++) {
    LINDXA[i] = LINDXAU[IWORK[i]];
    IWORK[i]  = i;
  }

  for(i = 0; i < N; i++) {
    /* search LINDXA such that    LINDXA[j]  == i */
    j = 0;
    do {
      fndd = (LINDXA[j] == i);
      j++;
    } while(!fndd);
    j--;
    /* search IWORK  such that    IWORK[k]   == j */
    k = 0;
    do {
      fndd = (IWORK[k] == j);
      k++;
    } while(!fndd);
    k--;
    /* swap IWORK[i] and IWORK[k]; LINDXAU[i] = k */
    j          = IWORK[i];
    IWORK[i]   = IWORK[k];
    IWORK[k]   = j;
    LINDXAU[i] = k;
  }
}
