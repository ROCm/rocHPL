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

void HPL_dlocmax(HPL_T_panel* PANEL,
                 const int    N,
                 const int    II,
                 const int    JJ,
                 double*      WORK,
                 int          thread_rank,
                 int          thread_size,
                 int*         max_index,
                 double*      max_value) {
  /*
   * Purpose
   * =======
   *
   * HPL_dlocmax finds  the maximum entry in the current column  and packs
   * the useful information in  WORK[0:3].  On exit,  WORK[0] contains the
   * local maximum  absolute value  scalar,  WORK[1] is the  corresponding
   * local row index,  WORK[2]  is the corresponding global row index, and
   * WORK[3] is the coordinate of the process owning this max.  When N  is
   * less than 1, the WORK[0:2] is initialized to zero, and WORK[3] is set
   * to the total number of process rows.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the local number of rows of the column
   *         of A on which we operate.
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
   *         On entry, WORK  is  a workarray of size at least 4.  On exit,
   *         WORK[0] contains  the  local  maximum  absolute value scalar,
   *         WORK[1] contains  the corresponding local row index,  WORK[2]
   *         contains the corresponding global row index, and  WORK[3]  is
   *         the coordinate of process owning this max.
   *
   * ---------------------------------------------------------------------
   */

  double* A;
  int     kk, igindx, ilindx, myrow, nb, nprow;

  if(N > 0) {
    A     = Mptr(PANEL->hA0, II, JJ, PANEL->lda0);
    myrow = PANEL->grid->myrow;
    nprow = PANEL->grid->nprow;
    nb    = PANEL->nb;

    HPL_idamax_omp(
        N, A, 1, nb, II, thread_rank, thread_size, max_index, max_value);

    if(thread_rank == 0) {
      ilindx = max_index[0];
      kk     = PANEL->ii + II + (ilindx);
      Mindxl2g(igindx, kk, nb, nb, myrow, 0, nprow);
      /*
       * WORK[0] := local maximum absolute value scalar,
       * WORK[1] := corresponding local  row index,
       * WORK[2] := corresponding global row index,
       * WORK[3] := coordinate of process owning this max.
       */
      WORK[0] = max_value[0];
      WORK[1] = (double)(ilindx);
      WORK[2] = (double)(igindx);
      WORK[3] = (double)(myrow);
    }
  } else {
    /*
     * If I do not have any row of A, then set the coordinate of the process
     * (WORK[3]) owning this "ghost" row,  such that it  will never be used,
     * even if there are only zeros in the current column of A.
     */
    if(thread_rank == 0) {
      WORK[0] = WORK[1] = WORK[2] = HPL_rzero;
      WORK[3]                     = (double)(PANEL->grid->nprow);
    }
  }

// make sure WORK is visible to all threads
#pragma omp barrier
}
