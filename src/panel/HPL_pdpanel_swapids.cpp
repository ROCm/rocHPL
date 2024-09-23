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

  if(jb <= 0) return;

  if(PANEL->grid->nprow == 1) {
    // unroll pivoting
    int* ipiv    = PANEL->ipiv;

    int* permU = PANEL->IWORK;
    int* upiv    = PANEL->IWORK + 2 * jb; // scratch space

    for(i = 0; i < jb; i++) { ipiv[i] -= PANEL->ii; } // shift
    HPL_unroll_ipiv(PANEL->mp, jb, ipiv, upiv, permU);

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
  }
}
