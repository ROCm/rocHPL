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

void HPL_pdpanel_SendToDevice(HPL_T_panel* PANEL) {
  const int jb = PANEL->jb;

  if((PANEL->grid->mycol != PANEL->pcol) || (jb <= 0)) return;

  if(PANEL->len > 0)
    CHECK_HIP_ERROR(hipMemcpyAsync(PANEL->A0,
                                   PANEL->hA0,
                                   PANEL->len * sizeof(double),
                                   hipMemcpyHostToDevice,
                                   dataStream));
}
