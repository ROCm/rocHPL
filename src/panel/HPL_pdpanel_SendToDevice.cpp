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

void HPL_unroll_ipiv(const int mp,
                     const int jb,
                     int*      ipiv,
                     int*      ipiv_ex,
                     int*      upiv) {

  for(int i = 0; i < mp; i++) { upiv[i] = i; } // initialize ids
  for(int i = 0; i < jb; i++) {                // swap ids
    int id        = upiv[i];
    upiv[i]       = upiv[ipiv[i]];
    upiv[ipiv[i]] = id;
  }

  for(int i = 0; i < jb; i++) { ipiv_ex[i] = -1; }

  int cnt = 0;
  for(int i = jb; i < mp; i++) { // find swapped ids outside of panel
    if(upiv[i] < jb) { ipiv_ex[upiv[i]] = i; }
  }
}

void HPL_pdpanel_SendToDevice(HPL_T_panel* PANEL) {
  int jb, i, ml2;

  jb = PANEL->jb;

  if(jb <= 0) return;

  // only the root column copies to device
  if(PANEL->grid->mycol == PANEL->pcol) {

    if(PANEL->grid->nprow == 1) {

      // unroll pivoting and send to device now
      int* ipiv    = PANEL->ipiv;
      int* ipiv_ex = PANEL->ipiv + jb;
      int* upiv    = PANEL->IWORK + jb; // scratch space

      for(i = 0; i < jb; i++) { ipiv[i] -= PANEL->ii; } // shift
      HPL_unroll_ipiv(PANEL->mp, jb, ipiv, ipiv_ex, upiv);

      for(i = 0; i < jb; i++) { ipiv[i] = upiv[i]; }

    } else {

      int  k       = (int)((unsigned int)(jb) << 1);
      int* iflag   = PANEL->IWORK;
      int* ipl     = iflag + 1;
      int* ipID    = ipl + 1;
      int* ipA     = ipID + ((unsigned int)(k) << 1);
      int* iplen   = ipA + 1;
      int* ipmap   = iplen + PANEL->grid->nprow + 1;
      int* ipmapm1 = ipmap + PANEL->grid->nprow;
      int* upiv    = ipmapm1 + PANEL->grid->nprow;
      int* iwork   = upiv + PANEL->mp;

      int* lindxU   = PANEL->lindxU;
      int* lindxA   = PANEL->lindxA;
      int* lindxAU  = PANEL->lindxAU;
      int* permU    = PANEL->permU;
      int* permU_ex = permU + jb;
      int* ipiv     = PANEL->ipiv;

      if(*iflag == -1) /* no index arrays have been computed so far */
      {
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
        *iflag = 1;
      }
    }
  }

  // copy A and/or L2
  if(PANEL->grid->mycol == PANEL->pcol) {
    if(PANEL->grid->npcol > 1) { // L2 is its own array
      if(PANEL->grid->myrow == PANEL->prow) {
        if((PANEL->mp - jb) > 0)
          CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->L2,
                                           PANEL->ldl2 * sizeof(double),
                                           Mptr(PANEL->A, jb, -jb, PANEL->lda),
                                           PANEL->lda * sizeof(double),
                                           (PANEL->mp - jb) * sizeof(double),
                                           jb,
                                           hipMemcpyDeviceToDevice,
                                           dataStream));
      } else {
        if((PANEL->mp) > 0)
          CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->L2,
                                           PANEL->ldl2 * sizeof(double),
                                           Mptr(PANEL->A, 0, -jb, PANEL->lda),
                                           PANEL->lda * sizeof(double),
                                           PANEL->mp * sizeof(double),
                                           jb,
                                           hipMemcpyDeviceToDevice,
                                           dataStream));
      }
    }
  }
}
