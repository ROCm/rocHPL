/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */
#include "hpl.hpp"

int HPL_pdwarmup(HPL_T_test* TEST,
                 HPL_T_grid* GRID,
                 HPL_T_palg* ALGO,
                 HPL_T_pmat* mat) {

  int N = mat->n;
  int NB = mat->nb;

  HPL_T_UPD_FUN HPL_pdupdate = ALGO->upfun;

  HPL_T_panel* p0 = &(mat->panel[0]);
  HPL_T_panel* p1 = &(mat->panel[1]);

  HPL_pdpanel_init(GRID, ALGO, N, N + 1, Mmin(N, NB), mat, 0, 0, MSGID_BEGIN_FACT, p0);
  HPL_pdpanel_init(GRID, ALGO, N, N + 1, Mmin(N, NB), mat, 0, 0, MSGID_BEGIN_FACT, p1);

  int mm = Mmin(p0->mp, p0->jb);
  int nn = Mmin(p0->nq, p0->jb);

  // Fill the matrix with values
  HPL_pdrandmat(GRID, N, N + 1, NB, mat->A, mat->ld, HPL_ISEED);

  // Do a pfact on all columns
  p0->pcol = p0->grid->mycol;
  HPL_pdpanel_SendToHost(p0);
  HPL_pdpanel_Wait(p0);
  for (int c=0;c<GRID->local_npcol;++c) {
    if (GRID->local_mycol==c) {
      HPL_pdfact(p0);
    }
    HPL_barrier(GRID->all_comm);
  }
  p0->A -= p0->jb * static_cast<size_t>(p0->lda);
  HPL_pdpanel_SendToDevice(p0);
  HPL_pdpanel_swapids(p0);
  HPL_pdpanel_Wait(p0);

  HPL_dlatcpy_gpu(mm,
                  nn,
                  p0->L1,
                  p0->jb,
                  p0->A,
                  p0->lda);

  //Broadcast to register with MPI
  p0->pcol = 0;
  HPL_pdpanel_bcast(p0);

  p0->nu0 = nn;
  p0->ldu0 = nn;
  HPL_pdlaswp_start(p0, HPL_LOOK_AHEAD);
  HPL_pdlaswp_exchange(p0, HPL_LOOK_AHEAD);
  HPL_pdlaswp_end(p0, HPL_LOOK_AHEAD);
  HPL_pdupdate(p0, HPL_LOOK_AHEAD);
  p0->nu0 = 0;

  HPL_pdlaswp_start(p0, HPL_UPD_1);
  HPL_pdlaswp_exchange(p0, HPL_UPD_1);
  HPL_pdlaswp_end(p0, HPL_UPD_1);
  HPL_pdupdate(p0, HPL_UPD_1);

  HPL_pdlaswp_start(p0, HPL_UPD_2);
  HPL_pdlaswp_exchange(p0, HPL_UPD_2);
  HPL_pdlaswp_end(p0, HPL_UPD_2);
  HPL_pdupdate(p0, HPL_UPD_2);

  CHECK_HIP_ERROR(hipDeviceSynchronize());

  // Do a pfact on all columns
  p1->pcol = p1->grid->mycol;
  HPL_pdpanel_SendToHost(p1);
  HPL_pdpanel_Wait(p1);
  for (int c=0;c<GRID->local_npcol;++c) {
    if (GRID->local_mycol==c) {
      HPL_pdfact(p1);
    }
    HPL_barrier(GRID->all_comm);
  }
  p1->A -= p1->jb * static_cast<size_t>(p1->lda);
  HPL_pdpanel_SendToDevice(p1);
  HPL_pdpanel_swapids(p1);
  HPL_pdpanel_Wait(p1);

  HPL_dlatcpy_gpu(mm,
                  nn,
                  p1->L1,
                  p1->jb,
                  p1->A,
                  p1->lda);

  //Broadcast to register with MPI
  p1->pcol = 0;
  HPL_pdpanel_bcast(p1);

  p1->nu0 = nn;
  p1->ldu0 = nn;
  HPL_pdlaswp_start(p1, HPL_LOOK_AHEAD);
  HPL_pdlaswp_exchange(p1, HPL_LOOK_AHEAD);
  HPL_pdlaswp_end(p1, HPL_LOOK_AHEAD);
  HPL_pdupdate(p1, HPL_LOOK_AHEAD);
  p1->nu0 = 0;

  HPL_pdlaswp_start(p1, HPL_UPD_1);
  HPL_pdlaswp_exchange(p1, HPL_UPD_1);
  HPL_pdlaswp_end(p1, HPL_UPD_1);
  HPL_pdupdate(p1, HPL_UPD_1);

  HPL_pdlaswp_start(p1, HPL_UPD_2);
  HPL_pdlaswp_exchange(p1, HPL_UPD_2);
  HPL_pdlaswp_end(p1, HPL_UPD_2);
  HPL_pdupdate(p1, HPL_UPD_2);

  HPL_pdtrsv(GRID, mat);

  return HPL_SUCCESS;
}
