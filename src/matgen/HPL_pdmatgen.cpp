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
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <unistd.h>

const int max_nthreads = 512;

static int Malloc(HPL_T_grid*  GRID,
                  void**       ptr,
                  const size_t bytes,
                  int          info[3]) {

  int mycol, myrow, npcol, nprow;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  unsigned long pg_size = sysconf(_SC_PAGESIZE);
  int           err     = posix_memalign(ptr, pg_size, bytes);

  /*Check allocation is valid*/
  info[0] = (err != 0);
  info[1] = myrow;
  info[2] = mycol;
  (void)HPL_all_reduce((void*)(info), 3, HPL_INT, HPL_MAX, GRID->all_comm);
  if(info[0] != 0) {
    return HPL_FAILURE;
  } else {
    return HPL_SUCCESS;
  }
}

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

int HPL_pdmatgen(HPL_T_test* TEST,
                 HPL_T_grid* GRID,
                 HPL_T_palg* ALGO,
                 HPL_T_pmat* mat,
                 const int   N,
                 const int   NB) {

  int ii, ip2, im4096;
  int mycol, myrow, npcol, nprow, nq, info[3];
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  int rank = GRID->iam;

  mat->n  = N;
  mat->nb = NB;
  mat->mp = HPL_numroc(N, NB, NB, myrow, 0, nprow);
  nq      = HPL_numroc(N, NB, NB, mycol, 0, npcol);
  /*
   * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
   * N by N+1.  One column is added in every process column for the solve.
   * The  result  however  is stored in a 1 x N vector replicated in every
   * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
   * workspace is mp.
   */
  mat->ld = Mmax(1, mat->mp);
  mat->ld = ((mat->ld + 95) / 128) * 128 + 32; /*pad*/

  mat->nq = nq + 1;

  mat->A = nullptr;
  mat->X = nullptr;

  mat->W0  = nullptr;
  mat->W1  = nullptr;
  mat->W2  = nullptr;

  mat->loc_workspace = nullptr;
  mat->max_workspace = nullptr;
  mat->dev_workspace = nullptr;
  mat->barrier_space = nullptr;
  mat->host_flag = nullptr;
  mat->host_workspace = nullptr;

  mat->panel[0].A0 = nullptr;
  mat->panel[0].U0 = nullptr;
  mat->panel[0].U1 = nullptr;
  mat->panel[0].U2 = nullptr;
  mat->panel[1].A0 = nullptr;
  mat->panel[1].U0 = nullptr;
  mat->panel[1].U1 = nullptr;
  mat->panel[1].U2 = nullptr;

  mat->panel[0].IWORK = nullptr;
  mat->panel[1].IWORK = nullptr;

  int dev = 0;
  CHECK_HIP_ERROR(hipGetDevice(&dev));

  hipDeviceProp_t props;
  CHECK_HIP_ERROR(hipGetDeviceProperties(&props, dev));
  mat->pfact_max_blocks = props.multiProcessorCount;

  /*
   * Allocate dynamic memory
   */

  // allocate on device
  size_t totalDeviceMem = 0;
  size_t numbytes = ((size_t)(mat->ld) * (size_t)(mat->nq)) * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Local matrix size       = %g GiBs\n",
           ((double)numbytes) / (1024 * 1024 * 1024));
  }
#endif

  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->A), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for A and b. "
                "Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  // Allocate panel workspaces
  int ierr = 0;
  ierr = HPL_pdpanel_new(TEST, GRID, ALGO, mat, &mat->panel[0], totalDeviceMem);
  if(ierr == HPL_FAILURE) return HPL_FAILURE;
  ierr = HPL_pdpanel_new(TEST, GRID, ALGO, mat, &mat->panel[1], totalDeviceMem);
  if(ierr == HPL_FAILURE) return HPL_FAILURE;

  // W spaces
  /*pdtrsv needs two vectors for B and W */
  /*pdlange needs nq and 1024 as workspace */
  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);
  numbytes = (NB * NB) * sizeof(double);
  numbytes = Mmax(2 * Anp * sizeof(double), numbytes);
  numbytes = Mmax(mat->nq * sizeof(double), numbytes);
  numbytes = Mmax(1024 * sizeof(double), numbytes);

  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->W0), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for W0 workspace. "
                "Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  int ldu2 = 0;
  if(nprow > 1) {
    const int NSplit = Mmax(0, ((((int)(mat->nq * ALGO->frac)) / NB) * NB));
    int       nu2    = Mmin(mat->nq, NSplit);
    ldu2             = ((nu2 + 95) / 128) * 128 + 32; /*pad*/

    int nu1  = mat->nq - nu2;
    int ldu1 = ((nu1 + 95) / 128) * 128 + 32; /*pad*/

    numbytes = (NB * ldu1) * sizeof(double);
    totalDeviceMem += numbytes;
    if(deviceMalloc(GRID, (void**)&(mat->W1), numbytes, info) != HPL_SUCCESS) {
      if(rank == 0) {
        HPL_pwarn(TEST->outfp,
                  __LINE__,
                  "HPL_pdmatgen",
                  "[%d,%d] Device memory allocation failed for W1 workspace. "
                  "Requested %g GiBs total. Test Skiped.",
                  info[1],
                  info[2],
                  ((double)totalDeviceMem) / (1024 * 1024 * 1024));
      }
      return HPL_FAILURE;
    }
  } else {
    int nu2 = mat->nq;
    ldu2    = ((nu2 + 95) / 128) * 128 + 32; /*pad*/
  }

  numbytes = (NB * ldu2) * sizeof(double);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->W2), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for W2 workspace. "
                "Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  // seperate space for X vector
  numbytes = mat->nq * sizeof(double);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->X), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for X. Requested %g "
                "GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  // Workspaces for pdfact
  numbytes = sizeof(int) * mat->pfact_max_blocks;
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->loc_workspace), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  numbytes = sizeof(double) * mat->pfact_max_blocks;
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->max_workspace), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  numbytes = sizeof(double) * (4 + 2 * NB + NB * mat->pfact_max_blocks );
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->dev_workspace), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }

  numbytes = sizeof(uint32_t) * 2;
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->barrier_space), numbytes, info) != HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    }
    return HPL_FAILURE;
  }
  mat->barrier_space[0] = 0;
  mat->barrier_space[1] = 0;

  /*we need 4 + 4*JB entries of scratch for pdfact */
  numbytes = sizeof(double) * 2 * (4 + 2 * NB);
  if(hostMalloc(GRID, (void**)&(mat->host_workspace), numbytes, info) !=
     HPL_SUCCESS) {
    if(rank == 0) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdmatgen",
                "[%d,%d] Host memory allocation failed for pfact workspace. "
                "Test Skiped.",
                info[1],
                info[2]);
    }
    return HPL_FAILURE;
  }

  numbytes = sizeof(int32_t);
  if(hostMalloc(GRID, (void**)&(mat->host_flag), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] Host memory allocation failed for pfact workspace. Test Skiped.",
              info[1],
              info[2]);
    return HPL_FAILURE;
  }
  mat->host_flag[0] = 0;

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Total device memory use = %g GiBs\n",
           ((double)totalDeviceMem) / (1024 * 1024 * 1024));
  }
#endif

  return HPL_SUCCESS;
}

void HPL_pdmatfree(HPL_T_pmat* mat) {

  if(mat->host_flag) {
    CHECK_HIP_ERROR(hipHostFree(mat->host_flag));
    mat->host_flag = nullptr;
  }
  if(mat->host_workspace) {
    CHECK_HIP_ERROR(hipHostFree(mat->host_workspace));
    mat->host_workspace = nullptr;
  }
  if(mat->barrier_space) {
    CHECK_HIP_ERROR(hipFree(mat->barrier_space));
    mat->barrier_space = nullptr;
  }
  if(mat->dev_workspace) {
    CHECK_HIP_ERROR(hipFree(mat->dev_workspace));
    mat->dev_workspace = nullptr;
  }
  if(mat->max_workspace) {
    CHECK_HIP_ERROR(hipFree(mat->max_workspace));
    mat->max_workspace = nullptr;
  }
  if(mat->loc_workspace) {
    CHECK_HIP_ERROR(hipFree(mat->loc_workspace));
    mat->loc_workspace = nullptr;
  }

  if(mat->X) {
    CHECK_HIP_ERROR(hipFree(mat->X));
    mat->X = nullptr;
  }
  if(mat->W2) {
    CHECK_HIP_ERROR(hipFree(mat->W2));
    mat->W2 = nullptr;
  }
  if(mat->W1) {
    CHECK_HIP_ERROR(hipFree(mat->W1));
    mat->W1 = nullptr;
  }
  if(mat->W0) {
    CHECK_HIP_ERROR(hipFree(mat->W0));
    mat->W0 = nullptr;
  }

  HPL_pdpanel_free(&(mat->panel[1]));
  HPL_pdpanel_free(&(mat->panel[0]));

  if(mat->A) {
    CHECK_HIP_ERROR(hipFree(mat->A));
    mat->A = nullptr;
  }
}
