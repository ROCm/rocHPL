/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#include "hpl.hpp"

void HPL_pdpanel_SendToHost(HPL_T_panel* PANEL) {
  int jb;

  jb = PANEL->jb;

  if((PANEL->grid->mycol != PANEL->pcol) || (jb <= 0)) return;

  if(PANEL->mp > 0)
    CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->A,
                                     PANEL->lda * sizeof(double),
                                     PANEL->dA,
                                     PANEL->dlda * sizeof(double),
                                     PANEL->mp * sizeof(double),
                                     jb,
                                     hipMemcpyDeviceToHost,
                                     dataStream));
}
