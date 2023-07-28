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

const int max_nthreads = 128;

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

  mat->dA = nullptr;
  mat->dX = nullptr;

  mat->dW = nullptr;
  mat->W  = nullptr;

  /* Create a rocBLAS handle */
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
  CHECK_ROCBLAS_ERROR(
      rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
  CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, computeStream));

  rocblas_initialize();

#ifdef HPL_ROCBLAS_ALLOW_ATOMICS
  CHECK_ROCBLAS_ERROR(
      rocblas_set_atomics_mode(handle, rocblas_atomics_allowed));
#else
  CHECK_ROCBLAS_ERROR(
      rocblas_set_atomics_mode(handle, rocblas_atomics_not_allowed));
#endif

  /*
   * Allocate dynamic memory
   */

  // allocate on device
  size_t numbytes = ((size_t)(mat->ld) * (size_t)(mat->nq)) * sizeof(double);

#ifdef HPL_VERBOSE_PRINT
  if((myrow == 0) && (mycol == 0)) {
    printf("Local matrix size = %g GBs\n",
           ((double)numbytes) / (1024 * 1024 * 1024));
  }
#endif

  if(deviceMalloc(GRID, (void**)&(mat->dA), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for for A and b. Skip.");
    return HPL_FAILURE;
  }

  // seperate space for X vector
  if(deviceMalloc(GRID, (void**)&(mat->dX), mat->nq * sizeof(double), info) !=
     HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for for x. Skip.");
    return HPL_FAILURE;
  }

  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);

  /*Need space for a column of panels for pdfact on CPU*/
  size_t A_hostsize = mat->ld * mat->nb * sizeof(double);

  if(hostMalloc(GRID, (void**)&(mat->A), A_hostsize, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Panel memory allocation failed. Skip.");
    return HPL_FAILURE;
  }

#pragma omp parallel
  {
    /*First touch*/
    const int thread_rank = omp_get_thread_num();
    const int thread_size = omp_get_num_threads();
    assert(thread_size <= max_nthreads);

    for(int i = 0; i < mat->ld; i += NB) {
      if((i / NB) % thread_size == thread_rank) {
        const int mm = std::min(NB, mat->ld - i);
        for(int k = 0; k < NB; ++k) {
          for(int j = 0; j < mm; ++j) {
            mat->A[j + i + static_cast<size_t>(mat->ld) * k] = 0.0;
          }
        }
      }
    }
  }

  size_t dworkspace_size = 0;
  size_t workspace_size  = 0;

  /*pdtrsv needs two vectors for B and W (and X on host) */
  dworkspace_size = Mmax(2 * Anp * sizeof(double), dworkspace_size);
  workspace_size  = Mmax((2 * Anp + nq) * sizeof(double), workspace_size);

  /*Scratch space for rows in pdlaswp (with extra space for padding) */
  dworkspace_size =
      Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), dworkspace_size);
  workspace_size =
      Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), workspace_size);

  if(deviceMalloc(GRID, (void**)&(mat->dW), dworkspace_size, info) !=
     HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for U workspace. Skip.");
    return HPL_FAILURE;
  }
  if(hostMalloc(GRID, (void**)&(mat->W), workspace_size, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Host memory allocation failed for U workspace. Skip.");
    return HPL_FAILURE;
  }

  return HPL_SUCCESS;
}

void HPL_pdmatfree(HPL_T_pmat* mat) {

  if(mat->dA) {
    CHECK_HIP_ERROR(hipFree(mat->dA));
    mat->dA = nullptr;
  }
  if(mat->dX) {
    CHECK_HIP_ERROR(hipFree(mat->dX));
    mat->dX = nullptr;
  }
  if(mat->dW) {
    CHECK_HIP_ERROR(hipFree(mat->dW));
    mat->dW = nullptr;
  }

  if(mat->A) {
    CHECK_HIP_ERROR(hipHostFree(mat->A));
    mat->A = nullptr;
  }
  if(mat->W) {
    CHECK_HIP_ERROR(hipHostFree(mat->W));
    mat->W = nullptr;
  }

  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
}
