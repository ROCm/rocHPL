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
  double *A, *dA;
  int     jb, i, ml2;

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

      int* dipiv    = PANEL->dipiv;
      int* dipiv_ex = PANEL->dipiv + jb;

      CHECK_HIP_ERROR(hipMemcpy2DAsync(dipiv,
                                       jb * sizeof(int),
                                       upiv,
                                       jb * sizeof(int),
                                       jb * sizeof(int),
                                       1,
                                       hipMemcpyHostToDevice,
                                       dataStream));
      CHECK_HIP_ERROR(hipMemcpy2DAsync(dipiv_ex,
                                       jb * sizeof(int),
                                       ipiv_ex,
                                       jb * sizeof(int),
                                       jb * sizeof(int),
                                       1,
                                       hipMemcpyHostToDevice,
                                       dataStream));

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

      int* dlindxU   = PANEL->dlindxU;
      int* dlindxA   = PANEL->dlindxA;
      int* dlindxAU  = PANEL->dlindxAU;
      int* dpermU    = PANEL->dpermU;
      int* dpermU_ex = dpermU + jb;
      int* dipiv     = PANEL->dipiv;

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

      int N = Mmax(*ipA, jb);
      if(N > 0) {
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dlindxA,
                                         k * sizeof(int),
                                         lindxA,
                                         k * sizeof(int),
                                         N * sizeof(int),
                                         1,
                                         hipMemcpyHostToDevice,
                                         dataStream));
        CHECK_HIP_ERROR(hipMemcpy2DAsync(dlindxAU,
                                         k * sizeof(int),
                                         lindxAU,
                                         k * sizeof(int),
                                         N * sizeof(int),
                                         1,
                                         hipMemcpyHostToDevice,
                                         dataStream));
      }

      CHECK_HIP_ERROR(hipMemcpyAsync(dlindxU,
                                     lindxU,
                                     jb * sizeof(int),
                                     hipMemcpyHostToDevice,
                                     dataStream));

      CHECK_HIP_ERROR(hipMemcpy2DAsync(dpermU,
                                       jb * sizeof(int),
                                       permU,
                                       jb * sizeof(int),
                                       jb * sizeof(int),
                                       1,
                                       hipMemcpyHostToDevice,
                                       dataStream));

      // send the ipivs along with L2 in the Bcast
      CHECK_HIP_ERROR(hipMemcpy2DAsync(dipiv,
                                       jb * sizeof(int),
                                       ipiv,
                                       jb * sizeof(int),
                                       jb * sizeof(int),
                                       1,
                                       hipMemcpyHostToDevice,
                                       dataStream));
    }
  }

  // copy A and/or L2
  if(PANEL->grid->mycol == PANEL->pcol) {
    // copy L1
    CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->dL1,
                                     jb * sizeof(double),
                                     PANEL->L1,
                                     jb * sizeof(double),
                                     jb * sizeof(double),
                                     jb,
                                     hipMemcpyHostToDevice,
                                     dataStream));

    if(PANEL->grid->npcol > 1) { // L2 is its own array
      if(PANEL->grid->myrow == PANEL->prow) {
        CHECK_HIP_ERROR(hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->dlda),
                                         PANEL->dlda * sizeof(double),
                                         Mptr(PANEL->A, 0, 0, PANEL->lda),
                                         PANEL->lda * sizeof(double),
                                         jb * sizeof(double),
                                         jb,
                                         hipMemcpyHostToDevice,
                                         dataStream));

        if((PANEL->mp - jb) > 0)
          CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->dL2,
                                           PANEL->dldl2 * sizeof(double),
                                           Mptr(PANEL->A, jb, 0, PANEL->lda),
                                           PANEL->lda * sizeof(double),
                                           (PANEL->mp - jb) * sizeof(double),
                                           jb,
                                           hipMemcpyHostToDevice,
                                           dataStream));
      } else {
        if((PANEL->mp) > 0)
          CHECK_HIP_ERROR(hipMemcpy2DAsync(PANEL->dL2,
                                           PANEL->dldl2 * sizeof(double),
                                           Mptr(PANEL->A, 0, 0, PANEL->lda),
                                           PANEL->lda * sizeof(double),
                                           PANEL->mp * sizeof(double),
                                           jb,
                                           hipMemcpyHostToDevice,
                                           dataStream));
      }
    } else {
      if(PANEL->mp > 0)
        CHECK_HIP_ERROR(hipMemcpy2DAsync(Mptr(PANEL->dA, 0, -jb, PANEL->dlda),
                                         PANEL->dlda * sizeof(double),
                                         Mptr(PANEL->A, 0, 0, PANEL->lda),
                                         PANEL->lda * sizeof(double),
                                         PANEL->mp * sizeof(double),
                                         jb,
                                         hipMemcpyHostToDevice,
                                         dataStream));
    }
  }
}
