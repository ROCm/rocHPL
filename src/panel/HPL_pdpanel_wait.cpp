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

void HPL_pdpanel_Wait(HPL_T_panel* PANEL) {

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_COPY);
#endif
  // Wait for panel
  CHECK_HIP_ERROR(hipStreamSynchronize(dataStream));
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_COPY);
#endif
}
