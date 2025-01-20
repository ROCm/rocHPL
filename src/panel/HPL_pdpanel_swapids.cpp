
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

void HPL_pdpanel_swapids(HPL_T_panel* PANEL) {
  int jb, i, ml2;

  jb = PANEL->jb;

  int nprow = PANEL->grid->nprow;

  if(jb <= 0) return;

  if(nprow == 1) {
    // unroll pivoting
    int* ipiv  = PANEL->ipiv;
    int* permU = PANEL->IWORK;
    int* ipl   = permU + 2 * jb;
    int* ipID  = ipl + 1;

    for(i = 0; i < jb; i++) permU[i + jb] = -1;

    HPL_pipid(PANEL, ipl, ipID);

    for(i = 0; i < *ipl; i += 2) {
      int src = ipID[i] - PANEL->ia;
      int dst = ipID[i + 1] - PANEL->ia;
      if(dst < jb) {
        permU[dst] = src;
      } else {
        permU[src + jb] = dst;
      }
    }

  } else {

    int* permU    = PANEL->IWORK;
    int* lindxU   = permU + jb;
    int* lindxA   = lindxU + jb;
    int* lindxAU  = lindxA + jb;

    int  k       = (int)((unsigned int)(jb) << 1);
    int* ipl     = lindxAU + jb;
    int* ipID    = ipl + 1;
    int* ipA     = ipID + ((unsigned int)(k) << 1);
    int* iplen   = ipA + 1;
    int* ipmap   = iplen + PANEL->grid->nprow + 1;
    int* ipmapm1 = ipmap + PANEL->grid->nprow;
    int* upiv    = ipmapm1 + PANEL->grid->nprow;
    int* iwork   = upiv + PANEL->mp;

    HPL_pipid(PANEL, ipl, ipID);
    HPL_plindx(
        PANEL, *ipl, ipID, ipA, lindxU, lindxAU, lindxA, iplen, permU, iwork);
  }
}
