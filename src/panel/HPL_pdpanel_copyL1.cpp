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

void HPL_pdpanel_copyL1(HPL_T_panel* PANEL) {
  const int jb = PANEL->jb;

  if((PANEL->grid->mycol != PANEL->pcol) ||
     (PANEL->grid->myrow != PANEL->prow) ||
     (jb <= 0))
    return;

  if (PANEL->algo->L1notran) {
    HPL_dlacpy_gpu(jb,
                   jb,
                   PANEL->L1,
                   jb,
                   Mptr(PANEL->A, 0, -jb, PANEL->lda),
                   PANEL->lda);
  } else {
    HPL_dlatcpy_gpu(jb,
                    jb,
                    PANEL->L1,
                    jb,
                    Mptr(PANEL->A, 0, -jb, PANEL->lda),
                    PANEL->lda);
  }
}
