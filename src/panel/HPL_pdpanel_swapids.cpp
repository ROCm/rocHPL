
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

void HPL_pdpanel_swapids(HPL_T_panel* PANEL) {
  int jb, i, ml2;

  jb = PANEL->jb;

  int nprow = PANEL->grid->nprow;

  if(jb <= 0) return;

  if(nprow == 1) {
    // unroll pivoting
    int* ipiv  = PANEL->ipiv;
    int* permU = PANEL->ipiv + jb;
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

    int* dpermU = PANEL->dipiv;

    // send pivoting ids to device
    CHECK_HIP_ERROR(hipMemcpyAsync(dpermU,
                                   permU,
                                   2 * jb * sizeof(int),
                                   hipMemcpyHostToDevice,
                                   dataStream));

  } else {

    int* permU    = PANEL->ipiv + jb;
    int* lindxU   = permU + jb;
    int* lindxA   = lindxU + jb;
    int* lindxAU  = lindxA + jb;
    int* ipA      = lindxAU + jb;
    int* iplen    = ipA + 1;

    int* ipl     = iplen + nprow + 1;
    int* ipID    = ipl + 1;
    int* iwork   = ipID + 4 * jb;

    HPL_pipid(PANEL, ipl, ipID);
    HPL_plindx(PANEL,
               *ipl,
               ipID,
               ipA,
               lindxU,
               lindxAU,
               lindxA,
               iplen,
               permU,
               iwork);

    int* dpermU = PANEL->dipiv;

    //send pivoting ids to device
    CHECK_HIP_ERROR(hipMemcpyAsync(dpermU,
                                   permU,
                                   (4 * jb + 1 + nprow + 1) * sizeof(int),
                                   hipMemcpyHostToDevice,
                                   dataStream));
  }
}
