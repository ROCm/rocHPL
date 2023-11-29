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

  mat->A = nullptr;
  mat->X = nullptr;

  mat->W = nullptr;

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

  if(deviceMalloc(GRID, (void**)&(mat->A), numbytes, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for A and b. Skip.");
    return HPL_FAILURE;
  }

  // seperate space for X vector
  if(deviceMalloc(GRID, (void**)&(mat->X), mat->nq * sizeof(double), info) !=
     HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_pdmatgen",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for x. Skip.");
    return HPL_FAILURE;
  }

  #pragma omp parallel
  {
    /*First touch*/
    const int thread_rank = omp_get_thread_num();
    const int thread_size = omp_get_num_threads();
    assert(thread_size <= max_nthreads);

    for(int nb = 0; nb < mat->nq; nb += NB) {
      for(int mb = nb; mb < mat->ld; mb += NB) {
        if(((mb-nb)/NB) % thread_size == thread_rank) {
          const int nn = std::min(NB, mat->nq - nb);
          const int mm = std::min(NB, mat->ld - mb);
          for(int j = 0; j < nn; ++j) {
            for(int i = 0; i < mm; i+=512) { // 4KB pages
              mat->A[(i + mb) + static_cast<size_t>(mat->ld) * (j + nb)] = 0.0;
            }
          }
        }
      }
    }
  }


  int Anp;
  Mnumroc(Anp, mat->n, mat->nb, mat->nb, myrow, 0, nprow);

  size_t workspace_size = 0;

  /*pdtrsv needs two vectors for B and W (and X on host) */
  workspace_size = Mmax(2 * Anp * sizeof(double), workspace_size);

  /*Scratch space for rows in pdlaswp (with extra space for padding) */
  workspace_size =
      Mmax((nq + mat->nb + 256) * mat->nb * sizeof(double), workspace_size);

  if(deviceMalloc(GRID, (void**)&(mat->W), workspace_size, info) !=
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

  return HPL_SUCCESS;
}

int HPL_WarmUp(HPL_T_test* TEST,
               HPL_T_grid* GRID,
               HPL_T_palg* ALGO,
               HPL_T_pmat* mat) {

  double target_warmup_time = 30.0; //seconds

  #ifdef HPL_VERBOSE_PRINT
    if((GRID->myrow == 0) && (GRID->mycol == 0)) {
      printf("Running warmup for %g seconds \n", target_warmup_time);
    }
  #endif

  int info[3];

  int NB = mat->nb;
  int mp = mat->mp;
  int nq = mat->nq-1;

  double *L, *U;

  int ldl = ((mp + 95) / 128) * 128 + 32; /*pad*/
  int ldu = ((nq + 95) / 128) * 128 + 32; /*pad*/

  int ml = mp + NB + 256; /*extra space for potential padding*/
  int nu = nq + NB + 256; /*extra space for potential padding*/

  size_t numbytesU = sizeof(double) * nu * NB;
  size_t numbytesL = sizeof(double) * ml * NB;
  if(deviceMalloc(GRID, (void**)&U, numbytesU, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_WarmUp",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for U. Skip.");
    return HPL_FAILURE;
  }
  if(deviceMalloc(GRID, (void**)&L, numbytesL, info) != HPL_SUCCESS) {
    HPL_pwarn(TEST->outfp,
              __LINE__,
              "HPL_WarmUp",
              "[%d,%d] %s",
              info[1],
              info[2],
              "Device memory allocation failed for L. Skip.");
    return HPL_FAILURE;
  }

  const double one  = 1.0;
  const double mone = -1.0;

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  int Niter = 10;

  CHECK_HIP_ERROR(hipEventRecord(dgemmStart[0], stream));
  for (int i=0;i<Niter;++i) {
    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_transpose,
                                      mp,
                                      nq,
                                      NB,
                                      &mone,
                                      L,
                                      ldl,
                                      U,
                                      ldu,
                                      &one,
                                      mat->A,
                                      mat->ld));
  }
  CHECK_HIP_ERROR(hipEventRecord(dgemmStop[0], stream));

  CHECK_HIP_ERROR(hipDeviceSynchronize());

  float gemmTime = 0.0;
  CHECK_HIP_ERROR(hipEventElapsedTime(&gemmTime, dgemmStart[0], dgemmStop[0]));
  gemmTime /= Niter;

  Niter = (target_warmup_time * 1000.0) / gemmTime;
  Niter = std::max(1, Niter);

  for (int i=0;i<Niter;++i) {
    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_transpose,
                                      mp,
                                      nq,
                                      NB,
                                      &mone,
                                      L,
                                      ldl,
                                      U,
                                      ldu,
                                      &one,
                                      mat->A,
                                      mat->ld));
  }

  CHECK_HIP_ERROR(hipDeviceSynchronize());

  CHECK_HIP_ERROR(hipFree(L));
  CHECK_HIP_ERROR(hipFree(U));

  return HPL_SUCCESS;
}

void HPL_pdmatfree(HPL_T_pmat* mat) {

  if(mat->A) {
    CHECK_HIP_ERROR(hipFree(mat->A));
    mat->A = nullptr;
  }
  if(mat->X) {
    CHECK_HIP_ERROR(hipFree(mat->X));
    mat->X = nullptr;
  }
  if(mat->W) {
    CHECK_HIP_ERROR(hipFree(mat->W));
    mat->W = nullptr;
  }

  CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
}
