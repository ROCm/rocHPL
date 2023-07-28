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

static int Malloc(HPL_T_grid* GRID, void** ptr, const size_t bytes) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  unsigned long pg_size = sysconf(_SC_PAGESIZE);
  int           err     = posix_memalign(ptr, pg_size, bytes);

  /*Check workspace allocation is valid*/
  if(err != 0) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

static int hostMalloc(HPL_T_grid* GRID, void** ptr, const size_t bytes) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  hipError_t err = hipHostMalloc(ptr, bytes);

  /*Check workspace allocation is valid*/
  if(err != hipSuccess) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

static int deviceMalloc(HPL_T_grid* GRID, void** ptr, const size_t bytes) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  hipError_t err = hipMalloc(ptr, bytes);

  /*Check workspace allocation is valid*/
  if(err != hipSuccess) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

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

  size_t dalign;
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
  PANEL->A  = A->A;
  PANEL->dA = Mptr((double*)(A->dA), ii, jj, A->ld);

  /*
   * Workspace pointers are initialized to NULL.
   */
  PANEL->L2    = nullptr;
  PANEL->dL2   = nullptr;
  PANEL->L1    = nullptr;
  PANEL->dL1   = nullptr;
  PANEL->DINFO = nullptr;
  PANEL->U     = nullptr;
  PANEL->dU    = nullptr;
  PANEL->W     = nullptr;
  PANEL->dW    = nullptr;
  PANEL->U1    = nullptr;
  PANEL->dU1   = nullptr;
  PANEL->W1    = nullptr;
  PANEL->dW1   = nullptr;
  PANEL->U2    = nullptr;
  PANEL->dU2   = nullptr;
  PANEL->W2    = nullptr;
  PANEL->dW2   = nullptr;
  // PANEL->WORK    = NULL;
  // PANEL->IWORK   = NULL;
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
  PANEL->dlda  = A->ld;   /* local leading dim of array A */
  PANEL->prow  = icurrow; /* proc row owning 1st row of trailing A */
  PANEL->pcol  = icurcol; /* proc col owning 1st col of trailing A */
  PANEL->msgid = TAG;     /* message id to be used for panel bcast */
                          /*
                           * Initialize  ldl2 and len to temporary dummy values and Update tag for
                           * next panel
                           */
  PANEL->ldl2  = 0;       /* local leading dim of array L2 */
  PANEL->dldl2 = 0;       /* local leading dim of array L2 */
  PANEL->len   = 0;       /* length of the buffer to broadcast */
  PANEL->nu0   = 0;
  PANEL->nu1   = 0;
  PANEL->nu2   = 0;
  PANEL->ldu0  = 0;
  PANEL->ldu1  = 0;
  PANEL->ldu2  = 0;

  /*
   * Figure out the exact amount of workspace  needed by the factorization
   * and the update - Allocate that space - Finish the panel data structu-
   * re initialization.
   *
   * L1:    JB x JB in all processes
   * DINFO: 1       in all processes
   *
   * We also make an array of necessary intergers for swaps in the update.
   *
   * If nprow is 1, we just allocate an array of 2*JB integers for the swap.
   * When nprow > 1, we allocate the space for the index arrays immediate-
   * ly. The exact size of this array depends on the swapping routine that
   * will be used, so we allocate the maximum:
   *
   *       lindxU   is of size         JB +
   *       lindxA   is of size at most JB +
   *       lindxAU  is of size at most JB +
   *       permU    is of size at most JB
   *
   *       ipiv     is of size at most JB
   *
   * that is  5*JB.
   *
   * We make sure that those three arrays are contiguous in memory for the
   * later panel broadcast (using type punning to put the integer array at
   * the end.  We  also  choose  to put this amount of space right after
   * L2 (when it exist) so that one can receive a contiguous buffer.
   */

  /*Split fraction*/
  const double fraction = ALGO->frac;

  dalign      = ALGO->align * sizeof(double);
  size_t lpiv = (5 * JB * sizeof(int) + sizeof(double) - 1) / (sizeof(double));

  if(npcol > 1) {
    ml2 = (myrow == icurrow ? mp - JB : mp);
    ml2 = Mmax(0, ml2);
    ml2 = ((ml2 + 95) / 128) * 128 + 32; /*pad*/
  } else {
    ml2 = 0; // L2 is aliased inside A
  }

  /* Size of LBcast message */
  PANEL->len = ml2 * JB + JB * JB + lpiv; // L2, L1, integer arrays

  /* space for L */
  lwork = PANEL->len + 1;

  nu  = Mmax(0, (mycol == icurcol ? nq - JB : nq));
  ldu = nu + JB + 256; /*extra space for potential padding*/

  /* space for U */
  uwork = JB * ldu;

  if(PANEL->max_lwork_size < (size_t)(lwork) * sizeof(double)) {
    if(PANEL->LWORK) {
      CHECK_HIP_ERROR(hipFree(PANEL->dLWORK));
      free(PANEL->LWORK);
    }
    // size_t numbytes = (((size_t)((size_t)(lwork) * sizeof( double )) +
    // (size_t)4095)/(size_t)4096)*(size_t)4096;
    size_t numbytes = (size_t)(lwork) * sizeof(double);

    if(deviceMalloc(GRID, (void**)&(PANEL->dLWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Device memory allocation failed for L workspace.");
    }
    if(hostMalloc(GRID, (void**)&(PANEL->LWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for L workspace.");
    }

    PANEL->max_lwork_size = (size_t)(lwork) * sizeof(double);
  }
  if(PANEL->max_uwork_size < (size_t)(uwork) * sizeof(double)) {
    if(PANEL->UWORK) {
      CHECK_HIP_ERROR(hipFree(PANEL->dUWORK));
      free(PANEL->UWORK);
    }
    // size_t numbytes = (((size_t)((size_t)(uwork) * sizeof( double )) +
    // (size_t)4095)/(size_t)4096)*(size_t)4096;
    size_t numbytes = (size_t)(uwork) * sizeof(double);

    if(deviceMalloc(GRID, (void**)&(PANEL->dUWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Device memory allocation failed for U workspace.");
    }
    if(hostMalloc(GRID, (void**)&(PANEL->UWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for U workspace.");
    }

    PANEL->max_uwork_size = (size_t)(uwork) * sizeof(double);
  }

  /*
   * Initialize the pointers of the panel structure
   */
  if(npcol == 1) {
    PANEL->L2    = PANEL->A + (myrow == icurrow ? JB : 0);
    PANEL->dL2   = PANEL->dA + (myrow == icurrow ? JB : 0);
    PANEL->ldl2  = A->ld;
    PANEL->dldl2 = A->ld; /*L2 is aliased inside A*/

    PANEL->L1  = (double*)PANEL->LWORK;
    PANEL->dL1 = (double*)PANEL->dLWORK;
  } else {
    PANEL->L2    = (double*)PANEL->LWORK;
    PANEL->dL2   = (double*)PANEL->dLWORK;
    PANEL->ldl2  = Mmax(0, ml2);
    PANEL->dldl2 = Mmax(0, ml2);

    PANEL->L1  = PANEL->L2 + ml2 * JB;
    PANEL->dL1 = PANEL->dL2 + ml2 * JB;
  }

  PANEL->U  = (double*)PANEL->UWORK;
  PANEL->dU = (double*)PANEL->dUWORK;
  PANEL->W  = A->W;
  PANEL->dW = A->dW;

  if(nprow == 1) {
    PANEL->nu0  = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    PANEL->ldu0 = PANEL->nu0;

    PANEL->nu1  = 0;
    PANEL->ldu1 = 0;

    PANEL->nu2  = nu - PANEL->nu0;
    PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

    PANEL->U2  = PANEL->U + JB * JB;
    PANEL->dU2 = PANEL->dU + JB * JB;
    PANEL->U1  = PANEL->U2 + PANEL->ldu2 * JB;
    PANEL->dU1 = PANEL->dU2 + PANEL->ldu2 * JB;

    PANEL->permU  = (int*)(PANEL->L1 + JB * JB);
    PANEL->dpermU = (int*)(PANEL->dL1 + JB * JB);
    PANEL->ipiv   = PANEL->permU + JB;
    PANEL->dipiv  = PANEL->dpermU + JB;

    PANEL->DINFO  = (double*)(PANEL->ipiv + 2 * JB);
    PANEL->dDINFO = (double*)(PANEL->dipiv + 2 * JB);
  } else {
    const int NSplit = Mmax(0, ((((int)(A->nq * fraction)) / nb) * nb));
    PANEL->nu0       = (mycol == inxtcol) ? Mmin(JB, nu) : 0;
    PANEL->ldu0      = PANEL->nu0;

    PANEL->nu2  = Mmin(nu - PANEL->nu0, NSplit);
    PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

    PANEL->nu1  = nu - PANEL->nu0 - PANEL->nu2;
    PANEL->ldu1 = ((PANEL->nu1 + 95) / 128) * 128 + 32; /*pad*/

    PANEL->U2  = PANEL->U + JB * JB;
    PANEL->dU2 = PANEL->dU + JB * JB;
    PANEL->U1  = PANEL->U2 + PANEL->ldu2 * JB;
    PANEL->dU1 = PANEL->dU2 + PANEL->ldu2 * JB;

    PANEL->W2  = PANEL->W + JB * JB;
    PANEL->dW2 = PANEL->dW + JB * JB;
    PANEL->W1  = PANEL->W2 + PANEL->ldu2 * JB;
    PANEL->dW1 = PANEL->dW2 + PANEL->ldu2 * JB;

    PANEL->lindxA   = (int*)(PANEL->L1 + JB * JB);
    PANEL->dlindxA  = (int*)(PANEL->dL1 + JB * JB);
    PANEL->lindxAU  = PANEL->lindxA + JB;
    PANEL->dlindxAU = PANEL->dlindxA + JB;
    PANEL->lindxU   = PANEL->lindxAU + JB;
    PANEL->dlindxU  = PANEL->dlindxAU + JB;
    PANEL->permU    = PANEL->lindxU + JB;
    PANEL->dpermU   = PANEL->dlindxU + JB;

    // Put ipiv array at the end
    PANEL->ipiv  = PANEL->permU + JB;
    PANEL->dipiv = PANEL->dpermU + JB;

    PANEL->DINFO  = ((double*)PANEL->lindxA) + lpiv;
    PANEL->dDINFO = ((double*)PANEL->dlindxA) + lpiv;
  }

  *(PANEL->DINFO) = 0.0;

  /*
   * If nprow is 1, we just allocate an array of JB integers to store the
   * pivot IDs during factoring, and a scratch array of mp integers.
   * When nprow > 1, we allocate the space for the index arrays immediate-
   * ly. The exact size of this array depends on the swapping routine that
   * will be used, so we allocate the maximum:
   *
   *    IWORK[0] is of size at most 1      +
   *    IPL      is of size at most 1      +
   *    IPID     is of size at most 4 * JB +
   *    IPIV     is of size at most JB     +
   *    SCRATCH  is of size at most MP
   *
   *    ipA      is of size at most 1      +
   *    iplen    is of size at most NPROW  + 1 +
   *    ipcounts is of size at most NPROW  +
   *    ioffsets is of size at most NPROW  +
   *    iwork    is of size at most MAX( 2*JB, NPROW+1 ).
   *
   * that is  mp + 4 + 5*JB + 3*NPROW + MAX( 2*JB, NPROW+1 ).
   *
   * We use the fist entry of this to work array  to indicate  whether the
   * the  local  index arrays have already been computed,  and if yes,  by
   * which function:
   *    IWORK[0] = -1: no index arrays have been computed so far;
   *    IWORK[0] =  1: HPL_pdlaswp already computed those arrays;
   * This allows to save some redundant and useless computations.
   */
  if(nprow == 1) {
    lwork = mp + JB;
  } else {
    itmp1 = (JB << 1);
    lwork = nprow + 1;
    itmp1 = Mmax(itmp1, lwork);
    lwork = mp + 4 + (5 * JB) + (3 * nprow) + itmp1;
  }

  if(PANEL->max_iwork_size < (size_t)(lwork) * sizeof(int)) {
    if(PANEL->IWORK) { free(PANEL->IWORK); }
    size_t numbytes = (size_t)(lwork) * sizeof(int);

    if(Malloc(GRID, (void**)&(PANEL->IWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for integer workspace.");
    }
    PANEL->max_iwork_size = (size_t)(lwork) * sizeof(int);
  }

  if(lwork) *(PANEL->IWORK) = -1;

  /*Finally, we need 4 + 4*JB entries of scratch for pdfact */
  lwork = (size_t)(((4 + ((unsigned int)(JB) << 1)) << 1));
  if(PANEL->max_fwork_size < (size_t)(lwork) * sizeof(double)) {
    if(PANEL->fWORK) { free(PANEL->fWORK); }
    size_t numbytes = (size_t)(lwork) * sizeof(double);

    if(Malloc(GRID, (void**)&(PANEL->fWORK), numbytes) != HPL_SUCCESS) {
      HPL_pabort(__LINE__,
                 "HPL_pdpanel_init",
                 "Host memory allocation failed for pdfact scratch workspace.");
    }
    PANEL->max_fwork_size = (size_t)(lwork) * sizeof(double);
  }
}
