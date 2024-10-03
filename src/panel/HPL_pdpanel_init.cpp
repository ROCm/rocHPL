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
#include <unistd.h>



void HPL_pdpanel_init(HPL_T_grid*  GRID,
                      HPL_T_palg*  ALGO,
                      const int    M,
                      const int    N,
                      const int    JB,
                      HPL_T_pmat*  A,
                      const int    IA,
                      const int    JA,
                      const int    TAG,
                      HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_init initializes a panel data structure.
   *
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * M       (local input)                 const int
   *         On entry, M specifies the global number of rows of the panel.
   *         M must be at least zero.
   *
   * N       (local input)                 const int
   *         On entry,  N  specifies  the  global number of columns of the
   *         panel and trailing submatrix. N must be at least zero.
   *
   * JB      (global input)                const int
   *         On entry, JB specifies is the number of columns of the panel.
   *         JB must be at least zero.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * IA      (global input)                const int
   *         On entry,  IA  is  the global row index identifying the panel
   *         and trailing submatrix. IA must be at least zero.
   *
   * JA      (global input)                const int
   *         On entry, JA is the global column index identifying the panel
   *         and trailing submatrix. JA must be at least zero.
   *
   * TAG     (global input)                const int
   *         On entry, TAG is the row broadcast message id.
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  int icurcol, icurrow, ii, itmp1, jj, lwork, uwork, ml2, mp, mycol, myrow, nb,
      npcol, nprow, nq, nu, ldu;

  PANEL->grid = GRID; /* ptr to the process grid */
  PANEL->algo = ALGO; /* ptr to the algo parameters */
  PANEL->pmat = A;    /* ptr to the local array info */

  myrow = GRID->myrow;
  mycol = GRID->mycol;
  nprow = GRID->nprow;
  npcol = GRID->npcol;
  nb    = A->nb;

  HPL_infog2l(IA,
              JA,
              nb,
              nb,
              nb,
              nb,
              0,
              0,
              myrow,
              mycol,
              nprow,
              npcol,
              &ii,
              &jj,
              &icurrow,
              &icurcol);
  mp = HPL_numrocI(M, IA, nb, nb, myrow, 0, nprow);
  nq = HPL_numrocI(N, JA, nb, nb, mycol, 0, npcol);

  const int inxtcol = MModAdd1(icurcol, npcol);
  const int inxtrow = MModAdd1(icurrow, nprow);

  /* ptr to trailing part of A */
  PANEL->A = Mptr((double*)(A->A), ii, jj, A->ld);

  /*
   * Local lengths, indexes process coordinates
   */
  PANEL->nb    = nb;      /* distribution blocking factor */
  PANEL->jb    = JB;      /* panel width */
  PANEL->m     = M;       /* global # of rows of trailing part of A */
  PANEL->n     = N;       /* global # of cols of trailing part of A */
  PANEL->ia    = IA;      /* global row index of trailing part of A */
  PANEL->ja    = JA;      /* global col index of trailing part of A */
  PANEL->mp    = mp;      /* local # of rows of trailing part of A */
  PANEL->nq    = nq;      /* local # of cols of trailing part of A */
  PANEL->ii    = ii;      /* local row index of trailing part of A */
  PANEL->jj    = jj;      /* local col index of trailing part of A */
  PANEL->lda   = A->ld;   /* local leading dim of array A */
  PANEL->prow  = icurrow; /* proc row owning 1st row of trailing A */
  PANEL->pcol  = icurcol; /* proc col owning 1st col of trailing A */
  PANEL->msgid = TAG;     /* message id to be used for panel bcast */
                          /*
                           * Initialize  ldl2 and len to temporary dummy values and Update tag for
                           * next panel
                           */
  PANEL->ldl2 = 0;        /* local leading dim of array L2 */
  PANEL->len  = 0;        /* length of the buffer to broadcast */
  PANEL->nu0  = 0;
  PANEL->nu1  = 0;
  PANEL->nu2  = 0;
  PANEL->ldu0 = 0;
  PANEL->ldu1 = 0;
  PANEL->ldu2 = 0;

  /*Split fraction*/
  const double fraction = ALGO->frac;

  size_t lpiv = ((4 * nb + 1 + nprow + 1) * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

  ml2 = mp;
  ml2 = Mmax(0, ml2);
  ml2 = ((ml2 + 95) / 128) * 128 + 32; /*pad*/

  /* Size of LBcast message */
  PANEL->len = ml2 * JB + JB * JB + lpiv; // L2, L1, integer arrays

  /*
   * Initialize the pointers of the panel structure
   */
  PANEL->lda0 = Mmax(0, ml2);
  PANEL->ldl2 = PANEL->lda0;
  PANEL->L2   = PANEL->A0 + (myrow == icurrow ? JB : 0);
  PANEL->L1   = PANEL->A0 + ml2 * JB;
  PANEL->dipiv = reinterpret_cast<int*>(PANEL->L1 + JB * JB);

  PANEL->ipiv = PANEL->IWORK;

  nu  = Mmax(0, (mycol == icurcol ? nq - JB : nq));
  ldu = nu + JB + 256; /*extra space for potential padding*/

  if(nprow == 1) {
    PANEL->nu0  = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    PANEL->ldu0 = PANEL->nu0;

    PANEL->nu1  = 0;
    PANEL->ldu1 = 0;

    PANEL->nu2  = nu - PANEL->nu0;
    PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/
  } else {
    const int NSplit = Mmax(0, ((((int)(A->nq * fraction)) / nb) * nb));
    PANEL->nu0       = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    PANEL->ldu0      = PANEL->nu0;

    PANEL->nu2  = Mmin(nu - PANEL->nu0, NSplit);
    PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

    PANEL->nu1  = nu - PANEL->nu0 - PANEL->nu2;
    PANEL->ldu1 = ((PANEL->nu1 + 95) / 128) * 128 + 32; /*pad*/
  }
}
