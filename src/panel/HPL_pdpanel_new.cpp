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

void HPL_pdpanel_new(HPL_T_grid*   GRID,
                     HPL_T_palg*   ALGO,
                     const int     M,
                     const int     N,
                     const int     JB,
                     HPL_T_pmat*   A,
                     const int     IA,
                     const int     JA,
                     const int     TAG,
                     HPL_T_panel** PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_new creates and initializes a panel data structure.
   *
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * M       (local input)                 const int
   *         On entry, M specifies the global number of rows of the panel.
   *         M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  global number of columns of the
   *         panel and trailing submatrix. N must be at least zero.
   *
   * JB      (global input)                const int
   *         On entry, JB specifies is the number of columns of the panel.
   *         JB must be at least zero.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * IA      (global input)                const int
   *         On entry,  IA  is  the global row index identifying the panel
   *         and trailing submatrix. IA must be at least zero.
   *
   * JA      (global input)                const int
   *         On entry, JA is the global column index identifying the panel
   *         and trailing submatrix. JA must be at least zero.
   *
   * TAG     (global input)                const int
   *         On entry, TAG is the row broadcast message id.
   *
   * PANEL   (local input/output)          HPL_T_panel * *
   *         On entry,  PANEL  points  to  the  address  of the panel data
   *         structure to create and initialize.
   *
   * ---------------------------------------------------------------------
   */

  HPL_T_panel* p = NULL;

  /*
   * Allocate the panel structure - Check for enough memory
   */
  if(!(p = (HPL_T_panel*)malloc(sizeof(HPL_T_panel)))) {
    HPL_pabort(__LINE__, "HPL_pdpanel_new", "Memory allocation failed");
  }

  p->max_pinned_work_size = 0;
  p->max_lwork_size       = 0;
  p->max_uwork_size       = 0;
  p->max_iwork_size       = 0;
  p->max_fwork_size       = 0;
  p->free_work_now        = 0;
  p->A                    = NULL;
  p->LWORK                = NULL;
  p->UWORK                = NULL;
  p->fWORK                = NULL;
  p->IWORK                = NULL;

  p->loc_workspace  = NULL;
  p->max_workspace  = NULL;
  p->dev_workspace  = NULL;
  p->host_flag      = NULL;
  p->host_workspace = NULL;
  p->locks          = NULL;

  HPL_pdpanel_init(GRID, ALGO, M, N, JB, A, IA, JA, TAG, p);
  *PANEL = p;
}
