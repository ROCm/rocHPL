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

  if(PANEL->pmat->info == 0) PANEL->pmat->info = *(PANEL->DINFO);

  if(PANEL->free_work_now == 1) {

    if(PANEL->LWORK) CHECK_HIP_ERROR(hipFree(PANEL->LWORK));
    if(PANEL->UWORK) CHECK_HIP_ERROR(hipFree(PANEL->UWORK));
    PANEL->max_lwork_size = 0;
    PANEL->max_uwork_size = 0;

    if(PANEL->IWORK) CHECK_HIP_ERROR(hipFree(PANEL->IWORK));
    if(PANEL->fWORK) CHECK_HIP_ERROR(hipFree(PANEL->fWORK));

    PANEL->max_iwork_size = 0;
    PANEL->max_fwork_size = 0;

    CHECK_HIP_ERROR(hipFree(PANEL->loc_workspace));
    CHECK_HIP_ERROR(hipFree(PANEL->max_workspace));
    CHECK_HIP_ERROR(hipFree(PANEL->dev_workspace));
    CHECK_HIP_ERROR(hipFree(PANEL->locks));

    CHECK_HIP_ERROR(hipHostFree(PANEL->host_flag));
    CHECK_HIP_ERROR(hipHostFree(PANEL->host_workspace));
  }

  return (HPL_SUCCESS);
}
