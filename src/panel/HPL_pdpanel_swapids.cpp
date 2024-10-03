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

static void HPL_unroll_ipiv(const int mp,
                            const int jb,
                            int*      ipiv,
                            int*      upiv,
                            int*      permU) {

  for(int i = 0; i < mp; i++) { upiv[i] = i; } // initialize ids
  for(int i = 0; i < jb; i++) {                // swap ids
    int id        = upiv[i];
    upiv[i]       = upiv[ipiv[i]];
    upiv[ipiv[i]] = id;
  }

  for(int i = 0; i < jb; i++) { permU[i+jb] = -1; }

  int cnt = 0;
  for(int i = jb; i < mp; i++) { // find swapped ids outside of panel
    if(upiv[i] < jb) { permU[upiv[i]+jb] = i; }
  }

  for(int i = 0; i < jb; i++) { permU[i] = upiv[i]; }
}

void HPL_pdpanel_swapids(HPL_T_panel* PANEL) {
  int jb, i, ml2;

  jb = PANEL->jb;

  int nprow = PANEL->grid->nprow;

  if(jb <= 0) return;

  if(nprow == 1) {
    // unroll pivoting
    int* ipiv    = PANEL->ipiv;

    int* permU = PANEL->ipiv + jb;
    int* upiv  = PANEL->ipiv + 3 * jb; // scratch space

    for(i = 0; i < jb; i++) { ipiv[i] -= PANEL->ii; } // shift
    HPL_unroll_ipiv(PANEL->mp, jb, ipiv, upiv, permU);

    int* dpermU = PANEL->dipiv;

    //send pivoting ids to device
    CHECK_HIP_ERROR(hipMemcpyAsync(dpermU,
                                   permU,
                                   2 * jb * sizeof(int),
                                   hipMemcpyHostToDevice,
                                   dataStream));
    CHECK_HIP_ERROR(hipStreamSynchronize(dataStream));

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
    CHECK_HIP_ERROR(hipStreamSynchronize(dataStream));
  }
}
