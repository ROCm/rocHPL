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

void HPL_dlocswpT(HPL_T_panel* PANEL,
                  const int    II,
                  const int    JJ,
                  double*      WORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlocswpT performs  the local swapping operations  within a panel.
   * The lower triangular  N0-by-N0  upper block of the panel is stored in
   * transpose form.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * II      (local input)                 const int
   *         On entry, II  specifies the row offset where the column to be
   *         operated on starts with respect to the panel.
   *
   * JJ      (local input)                 const int
   *         On entry, JJ  specifies the column offset where the column to
   *         be operated on starts with respect to the panel.
   *
   * WORK    (local workspace)             double *
   *         On entry, WORK  is a workarray of size at least 2 * (4+2*N0).
   *         WORK[0] contains  the  local  maximum  absolute value scalar,
   *         WORK[1] contains  the corresponding local row index,  WORK[2]
   *         contains the corresponding global row index, and  WORK[3]  is
   *         the coordinate of process owning this max.  The N0 length max
   *         row is stored in WORK[4:4+N0-1];  Note  that this is also the
   *         JJth row  (or column) of L1. The remaining part of this array
   *         is used as workspace.
   *
   * ---------------------------------------------------------------------
   */

  double  gmax;
  double *A1, *A2, *L, *Wr0, *Wmx;
  int     ilindx, lda, myrow, n0;

  myrow  = PANEL->grid->myrow;
  n0     = PANEL->jb;
  int NB = PANEL->nb;
  lda    = PANEL->lda0;

  Wr0     = (Wmx = WORK + 4) + NB;
  Wmx[JJ] = gmax = WORK[0];

  /*
   * Replicated swap and copy of the current (new) row of A into L1
   */
  L = Mptr(PANEL->hL1, 0, JJ, n0);
  /*
   * If the pivot is non-zero ...
   */
  if(gmax != HPL_rzero) {
    /*
     * and if I own the current row of A ...
     */
    if(myrow == PANEL->prow) {
      /*
       * and if I also own the row to be swapped with the current row of A ...
       */
      if(myrow == (int)(WORK[3])) {
        /*
         * and if the current row of A is not to swapped with itself ...
         */
        if((ilindx = (int)(WORK[1])) != 0) {
          /*
           * then copy the max row into L1 and locally swap the 2 rows of A.
           */
          A1 = Mptr(PANEL->hA0, II, 0, lda);
          A2 = Mptr(A1, ilindx, 0, lda);

          HPL_dcopy(n0, Wmx, 1, L, 1);
          HPL_dcopy(n0, Wmx, 1, A1, lda);
          HPL_dcopy(n0, Wr0, 1, A2, lda);

        } else {
          /*
           * otherwise the current row of  A  is swapped with itself, so just
           * copy the current of A into L1.
           */
          *Mptr(PANEL->hA0, II, JJ, lda) = gmax;

          HPL_dcopy(n0, Wmx, 1, L, 1);
        }

      } else {
        /*
         * otherwise, the row to be swapped with the current row of A is in Wmx,
         * so copy Wmx into L1 and A.
         */
        A1 = Mptr(PANEL->hA0, II, 0, lda);

        HPL_dcopy(n0, Wmx, 1, L, 1);
        HPL_dcopy(n0, Wmx, 1, A1, lda);
      }

    } else {
      /*
       * otherwise I do not own the current row of A, so copy the max row  Wmx
       * into L1.
       */
      HPL_dcopy(n0, Wmx, 1, L, 1);

      /*
       * and if I own the max row, overwrite it with the current row Wr0.
       */
      if(myrow == (int)(WORK[3])) {
        A2 = Mptr(PANEL->hA0, II + (size_t)(WORK[1]), 0, lda);

        HPL_dcopy(n0, Wr0, 1, A2, lda);
      }
    }
  } else {
    /*
     * Otherwise the max element in the current column is zero,  simply copy
     * the current row Wr0 into L1. The matrix is singular.
     */
    HPL_dcopy(n0, Wr0, 1, L, 1);
  }
}
