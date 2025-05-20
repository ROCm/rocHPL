/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    HPL - 2.2 - February 24, 2016
 *    Antoine P. Petitet
 *    University of Tennessee, Knoxville
 *    Innovative Computing Laboratory
 *    (C) Copyright 2000-2008 All Rights Reserved
 *
 *    Modified by: Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"

static int hostMalloc(HPL_T_grid*  GRID,
                      void**       ptr,
                      const size_t bytes,
                      int          info[3]) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  hipError_t err = hipHostMalloc(ptr, bytes);

  /*Check allocation is valid*/
  info[0] = (err != hipSuccess);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

static int deviceMalloc(HPL_T_grid*  GRID,
                        void**       ptr,
                        const size_t bytes,
                        int          info[3]) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  hipError_t err = hipMalloc(ptr, bytes);

  /*Check allocation is valid*/
  info[0] = (err != hipSuccess);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

int HPL_pdpanel_new(HPL_T_test*  TEST,
                    HPL_T_grid*  GRID,
                    HPL_T_palg*  ALGO,
                    HPL_T_pmat*  A,
                    HPL_T_panel* PANEL,
                    size_t&      totalMem) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_new creates and initializes a panel data structure.
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
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * PANEL   (local input/output)          HPL_T_panel * *
   *         On entry,  PANEL  points  to  the  address  of the panel data
   *         structure to create and initialize.
   *
   * ---------------------------------------------------------------------
   */

  int info[3];

  int myrow = GRID->myrow;
  int mycol = GRID->mycol;
  int nprow = GRID->nprow;
  int npcol = GRID->npcol;
  int rank  = GRID->iam;
  int N     = A->n;
  int nb    = A->nb;

  // Local number of rows and columns
  int mp = HPL_numrocI(N, 0, nb, nb, myrow, 0, nprow);
  int nq = HPL_numrocI(N + 1, 0, nb, nb, mycol, 0, npcol);

  // LBroadcast Space. Holds A0/L2 + L1 + pivoting arrays
  size_t lpiv = ((4 * nb + 1 + nprow + 1) * sizeof(int) + sizeof(double) - 1) /
                (sizeof(double));
  size_t numbytes = (A->ld * nb + nb * nb + lpiv) * sizeof(double);

  totalMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(PANEL->A0), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdpanel_new",
                "[%d,%d] Device memory allocation failed for Panel L "
                "workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  // U spaces
  numbytes = (nb * nb) * sizeof(double);

  totalMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(PANEL->U0), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdpanel_new",
                "[%d,%d] Device memory allocation failed for Panel U0 "
                "workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  if(nprow > 1) {
    int Anq=0;
    Mnumroc(Anq, A->n+1, nb, nb, mycol, 0, npcol);
    const int NSplit1 = Mmax(0, ((((int)(Anq * ALGO->frac)) / nb) * nb));
    const int NSplit2 = Mmax(0, Anq - NSplit1);

    PANEL->nu2       = Mmin(Anq, NSplit2);
    PANEL->ldu2      = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/

    PANEL->nu1  = Anq - PANEL->nu2;
    PANEL->ldu1 = ((PANEL->nu1 + 95) / 128) * 128 + 32; /*pad*/

    numbytes = (nb * PANEL->ldu1) * sizeof(double);
    totalMem += numbytes;
    if(deviceMalloc(GRID, (void**)&(PANEL->U1), numbytes, info) !=
       HPL_SUCCESS) {
      if(rank == 0) {
        HPL_pwarn(TEST->outfp,
                  __LINE__,
                  "HPL_pdpanel_new",
                  "[%d,%d] Device memory allocation failed for Panel U1 "
                  "workspace. Requested %g GiBs total. Test Skiped.",
                  info[1],
                  info[2],
                  ((double)totalMem) / (1024 * 1024 * 1024));
      }
      return HPL_FAILURE;
    }
  } else {
    PANEL->nu2  = A->nq;
    PANEL->ldu2 = ((PANEL->nu2 + 95) / 128) * 128 + 32; /*pad*/
  }

  numbytes = (nb * PANEL->ldu2) * sizeof(double);
  totalMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(PANEL->U2), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdpanel_new",
                "[%d,%d] Device memory allocation failed for Panel U2 "
                "workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  /*
   * If nprow is 1, we just allocate an array of NB integers to store the
   * pivot IDs during factoring, and a scratch array of mp integers.
   * When nprow > 1, we allocate the space for the index arrays immediate-
   * ly. The exact size of this array depends on the swapping routine that
   * will be used, so we allocate the maximum:
   *
   *    permU    is of size at most NB     +
   *    lindxA   is of size at most NB     +
   *    lindxAU  is of size at most NB     +
   *    lindxU   is of size at most NB     +
   *
   *    IPL      is of size at most 1      +
   *    IPID     is of size at most 4 * NB +
   *    IPIV     is of size at most NB     +
   *    SCRATCH  is of size at most MP
   *
   *    ipA      is of size at most 1      +
   *    iplen    is of size at most NPROW  + 1 +
   *    ipcounts is of size at most NPROW  +
   *    ioffsets is of size at most NPROW  +
   *    iwork    is of size at most MAX( 2*NB, NPROW+1 ).
   *
   * that is  mp + 3 + 9*NB + 3*NPROW + MAX( 2*NB, NPROW+1 ).
   */

  size_t itmp1 = (nb << 1);
  size_t iwork = nprow + 1;
  itmp1        = Mmax(itmp1, iwork);
  iwork        = mp + 3 + (9 * nb) + (3 * nprow) + itmp1;

  numbytes = iwork * sizeof(int);
  if(hostMalloc(GRID, (void**)&(PANEL->IWORK), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdpanel_new",
                "[%d,%d] Host memory allocation failed for integer workspace. "
                "Test Skiped.",
                info[1],
                info[2]);
    }
    return HPL_FAILURE;
  }

  return HPL_SUCCESS;
}
