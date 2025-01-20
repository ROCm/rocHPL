/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#include "hpl.hpp"

void HPL_pdpanel_SendToHost(HPL_T_panel* PANEL) {
  const int jb = PANEL->jb;

  if((PANEL->grid->mycol != PANEL->pcol) || (jb <= 0)) return;

  HPL_dlacpy_gpu(PANEL->mp,
                 PANEL->jb,
                 PANEL->A,
                 PANEL->lda,
                 PANEL->A0,
                 PANEL->lda0);

  CHECK_HIP_ERROR(hipEventRecord(panelCopy, computeStream));
}
