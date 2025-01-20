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

int HPL_pdpanel_free(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_free deallocates  the panel resources  and  stores the error
   * code returned by the panel factorization.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points  to  the  panel data  structure from
   *         which the resources should be deallocated.
   *
   * ---------------------------------------------------------------------
   */

  if(PANEL->IWORK) {
    CHECK_HIP_ERROR(hipFree(PANEL->IWORK));
    PANEL->IWORK = nullptr;
  }
  if(PANEL->U2) {
    CHECK_HIP_ERROR(hipFree(PANEL->U2));
    PANEL->U2 = nullptr;
  }
  if(PANEL->U1) {
    CHECK_HIP_ERROR(hipFree(PANEL->U1));
    PANEL->U1 = nullptr;
  }
  if(PANEL->U0) {
    CHECK_HIP_ERROR(hipFree(PANEL->U0));
    PANEL->U0 = nullptr;
  }
  if(PANEL->A0) {
    CHECK_HIP_ERROR(hipFree(PANEL->A0));
    PANEL->A0 = nullptr;
  }

  return (HPL_SUCCESS);
}
