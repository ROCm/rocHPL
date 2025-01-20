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

void HPL_pdpanel_Wait(HPL_T_panel* PANEL) {

  // Wait for panel
  CHECK_HIP_ERROR(hipEventSynchronize(panelCopy));
}
