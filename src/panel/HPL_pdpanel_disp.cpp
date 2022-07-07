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

int HPL_pdpanel_disp(HPL_T_panel** PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_disp deallocates  the  panel  structure  and  resources  and
   * stores the error code returned by the panel factorization.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel * *
   *         On entry,  PANEL  points  to  the  address  of the panel data
   *         structure to be deallocated.
   *
   * ---------------------------------------------------------------------
   */

  int mpierr;

  /*
   * Deallocate the panel resources and panel structure
   */
  (*PANEL)->free_work_now = 1;
  mpierr                  = HPL_pdpanel_free(*PANEL);
  if(*PANEL) free(*PANEL);
  *PANEL = NULL;

  return (mpierr);
}
