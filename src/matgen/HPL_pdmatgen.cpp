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
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <unistd.h>

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

  mat->n    = N;
  mat->nb   = NB;
  mat->info = 0;
  mat->mp   = HPL_numroc(N, NB, NB, myrow, 0, nprow);
  nq        = HPL_numroc(N, NB, NB, mycol, 0, npcol);
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

  mat->A  = nullptr;
  mat->X  = nullptr;
  mat->W0 = nullptr;
  mat->W1 = nullptr;
  mat->W2 = nullptr;

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

  mat->loc_workspace = nullptr;
  mat->max_workspace = nullptr;
  mat->dev_workspace = nullptr;
  mat->locks = nullptr;
  mat->host_flag = nullptr;
  mat->host_workspace = nullptr;

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
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for A and b. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  // Allocate panel workspaces
  int ierr = 0;
  ierr = HPL_pdpanel_new(TEST, GRID, ALGO, mat, &mat->panel[0], totalDeviceMem);
  if (ierr == HPL_FAILURE) return HPL_FAILURE;
  ierr = HPL_pdpanel_new(TEST, GRID, ALGO, mat, &mat->panel[1], totalDeviceMem);
  if (ierr == HPL_FAILURE) return HPL_FAILURE;

  // W spaces
  /*pdtrsv needs two vectors for B and W */
  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);
  numbytes = (NB * NB) * sizeof(double);
  numbytes = Mmax(2 * Anp * sizeof(double), numbytes);

  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->W0), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for W0 workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  int ldu2 = 0;
  if(nprow > 1) {
    const int NSplit = Mmax(0, ((((int)(mat->nq * ALGO->frac)) / NB) * NB));
    int nu2 = Mmin(mat->nq, NSplit);
    ldu2 = ((nu2 + 95) / 128) * 128 + 32; /*pad*/

    int nu1  = mat->nq - nu2;
    int ldu1 = ((nu1 + 95) / 128) * 128 + 32; /*pad*/

    numbytes = (NB * ldu1) * sizeof(double);
    totalDeviceMem += numbytes;
    if(deviceMalloc(GRID, (void**)&(mat->W1), numbytes, info) != HPL_SUCCESS) {
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdgenerate",
                "[%d,%d] Device memory allocation failed for W1 workspace. Requested %g GiBs total. Test Skiped.",
                info[1],
                info[2],
                ((double)totalDeviceMem) / (1024 * 1024 * 1024));
      return HPL_FAILURE;
    }
  } else {
    int nu2 = mat->nq;
    ldu2 = ((nu2 + 95) / 128) * 128 + 32; /*pad*/
  }

  numbytes = (NB * ldu2) * sizeof(double);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->W2), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for W2 workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }


  // seperate space for X vector
  numbytes = mat->nq * sizeof(double);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->X), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for X. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }


  // Workspaces for pdfact
  numbytes = sizeof(int) * ((mat->mp + NB-1)/NB + 1);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->loc_workspace), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  numbytes = sizeof(double) * ((mat->mp + NB-1)/NB + 1);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->max_workspace), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  numbytes = sizeof(double) * (NB * ((mat->mp + NB-1)/NB + 1) );
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->dev_workspace), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  numbytes = sizeof(uint32_t) * 2;
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->locks), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }
  mat->locks[0] = 0;
  mat->locks[1] = 0;

  /*we need 4 + 4*JB entries of scratch for pdfact */
  numbytes = sizeof(double) * 2 * (4 + 2 * NB);
  totalDeviceMem += numbytes;
  if(deviceMalloc(GRID, (void**)&(mat->host_workspace), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
              "[%d,%d] Device memory allocation failed for pfact workspace. Requested %g GiBs total. Test Skiped.",
              info[1],
              info[2],
              ((double)totalDeviceMem) / (1024 * 1024 * 1024));
    return HPL_FAILURE;
  }

  if(hostMalloc(GRID, (void**)&(mat->host_flag), sizeof(int32_t), info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdgenerate",
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

  // #pragma omp parallel
  // {
  //   /*First touch*/
  //   const int thread_rank = omp_get_thread_num();
  //   const int thread_size = omp_get_num_threads();
  //   assert(thread_size <= max_nthreads);

  //   for(int nb = 0; nb < mat->nq; nb += NB) {
  //     for(int mb = nb; mb < mat->ld; mb += NB) {
  //       if(((mb-nb)/NB) % thread_size == thread_rank) {
  //         const int nn = std::min(NB, mat->nq - nb);
  //         const int mm = std::min(NB, mat->ld - mb);
  //         for(int j = 0; j < nn; ++j) {
  //           for(int i = 0; i < mm; i+=512) { // 4KB pages
  //             mat->A[(i + mb) + static_cast<size_t>(mat->ld) * (j + nb)] = 0.0;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  return HPL_SUCCESS;
}

int HPL_WarmUp(HPL_T_test* TEST,
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
  HPL_dlacpy_gpu(mm,
                 nn,
                 p0->A,
                 p0->lda,
                 p0->A0,
                 p0->lda0);

  p0->pcol = p0->grid->mycol;
  HPL_pdfact(p0);
  p0->A -= p0->jb * static_cast<size_t>(p0->lda);

  HPL_dlatcpy_gpu(mm,
                  nn,
                  p0->L1,
                  p0->jb,
                  p0->A,
                  p0->lda);

  //Broadcast to register with MPI
  p0->pcol = 0;
  HPL_pdpanel_bcast(p0);
  HPL_pdpanel_swapids(p0);

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

  // Do a pfact on all columns
  HPL_dlacpy_gpu(mm,
                 nn,
                 p1->A,
                 p1->lda,
                 p1->A0,
                 p1->lda0);

  p1->pcol = p1->grid->mycol;
  HPL_pdfact(p1);
  p1->A -= p1->jb * static_cast<size_t>(p1->lda);

  HPL_dlatcpy_gpu(mm,
                  nn,
                  p1->L1,
                  p1->jb,
                  p1->A,
                  p1->lda);

  //Broadcast to register with MPI
  p1->pcol = 0;
  HPL_pdpanel_bcast(p1);
  HPL_pdpanel_swapids(p1);

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

void HPL_pdmatfree(HPL_T_pmat* mat) {

if(mat->host_flag) {
    CHECK_HIP_ERROR(hipHostFree(mat->host_flag));
    mat->host_flag = nullptr;
  }
  if(mat->host_workspace) {
    CHECK_HIP_ERROR(hipFree(mat->host_workspace));
    mat->host_workspace = nullptr;
  }
  if(mat->locks) {
    CHECK_HIP_ERROR(hipFree(mat->locks));
    mat->locks = nullptr;
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
