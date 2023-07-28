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
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 512
#define GRID_SIZE 512

__global__ void normA_1(const int N,
                        const int M,
                        const double* __restrict__ A,
                        const int LDA,
                        double* __restrict__ normAtmp) {
  __shared__ double s_norm[BLOCK_SIZE];

  const int t  = threadIdx.x;
  const int i  = blockIdx.x;
  size_t    id = i * BLOCK_SIZE + t;

  s_norm[t] = 0.0;
  for(; id < (size_t)N * M; id += gridDim.x * BLOCK_SIZE) {
    const int    m   = id % M;
    const int    n   = id / M;
    const double Anm = fabs(A[n + ((size_t)m) * LDA]);

    s_norm[t] = (Anm > s_norm[t]) ? Anm : s_norm[t];
  }
  __syncthreads();

  for(int k = BLOCK_SIZE / 2; k > 0; k /= 2) {
    if(t < k) {
      s_norm[t] = (s_norm[t + k] > s_norm[t]) ? s_norm[t + k] : s_norm[t];
    }
    __syncthreads();
  }

  if(t == 0) normAtmp[i] = s_norm[0];
}

__global__ void normA_2(const int N, double* __restrict__ normAtmp) {
  __shared__ double s_norm[BLOCK_SIZE];

  const int t = threadIdx.x;

  s_norm[t] = 0.0;
  for(size_t id = t; id < N; id += BLOCK_SIZE) {
    const double Anm = normAtmp[id];
    s_norm[t]        = (Anm > s_norm[t]) ? Anm : s_norm[t];
  }
  __syncthreads();

  for(int k = BLOCK_SIZE / 2; k > 0; k /= 2) {
    if(t < k) {
      s_norm[t] = (s_norm[t + k] > s_norm[t]) ? s_norm[t + k] : s_norm[t];
    }
    __syncthreads();
  }

  if(t == 0) normAtmp[0] = s_norm[0];
}

__global__ void norm1(const int N,
                      const int M,
                      const double* __restrict__ A,
                      const int LDA,
                      double* __restrict__ work) {

  __shared__ double s_norm1[BLOCK_SIZE];

  const int t = threadIdx.x;
  const int n = blockIdx.x;

  s_norm1[t] = 0.0;
  for(size_t id = t; id < M; id += BLOCK_SIZE) {
    s_norm1[t] += fabs(A[id + n * ((size_t)LDA)]);
  }

  __syncthreads();

  for(int k = BLOCK_SIZE / 2; k > 0; k /= 2) {
    if(t < k) { s_norm1[t] += s_norm1[t + k]; }
    __syncthreads();
  }

  if(t == 0) work[n] = s_norm1[0];
}

__global__ void norminf(const int N,
                        const int M,
                        const double* __restrict__ A,
                        const int LDA,
                        double* __restrict__ work) {
  const int    t  = threadIdx.x;
  const int    b  = blockIdx.x;
  const size_t id = b * BLOCK_SIZE + t; // row id

  if(id < M) {
    double norm = 0.0;
    for(size_t i = 0; i < N; i++) { norm += fabs(A[id + i * ((size_t)LDA)]); }
    work[id] = norm;
  }
}

double HPL_pdlange(const HPL_T_grid* GRID,
                   const HPL_T_NORM  NORM,
                   const int         M,
                   const int         N,
                   const int         NB,
                   const double*     A,
                   const int         LDA) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlange returns  the value of the one norm,  or the infinity norm,
   * or the element of largest absolute value of a distributed matrix A:
   *
   *
   *    max(abs(A(i,j))) when NORM = HPL_NORM_A,
   *    norm1(A),        when NORM = HPL_NORM_1,
   *    normI(A),        when NORM = HPL_NORM_I,
   *
   * where norm1 denotes the one norm of a matrix (maximum column sum) and
   * normI denotes  the infinity norm of a matrix (maximum row sum).  Note
   * that max(abs(A(i,j))) is not a matrix norm.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * NORM    (global input)                const HPL_T_NORM
   *         On entry,  NORM  specifies  the  value to be returned by this
   *         function as described above.
   *
   * M       (global input)                const int
   *         On entry,  M  specifies  the number  of rows of the matrix A.
   *         M must be at least zero.
   *
   * N       (global input)                const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix. NB must be larger than one.
   *
   * A       (local input)                 const double *
   *         On entry,  A  points to an array of dimension  (LDA,LocQ(N)),
   *         that contains the local pieces of the distributed matrix A.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,LocP(M)).
   *
   * ---------------------------------------------------------------------
   */

  double   s, v0 = HPL_rzero, *work = NULL, *dwork = NULL;
  MPI_Comm Acomm, Ccomm, Rcomm;
  int      ii, jj, mp, mycol, myrow, npcol, nprow, nq;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);
  Rcomm = GRID->row_comm;
  Ccomm = GRID->col_comm;
  Acomm = GRID->all_comm;

  Mnumroc(mp, M, NB, NB, myrow, 0, nprow);
  Mnumroc(nq, N, NB, NB, mycol, 0, npcol);

  if(Mmin(M, N) == 0) {
    return (v0);
  } else if(NORM == HPL_NORM_A) {
    /*
     * max( abs( A ) )
     */
    if((nq > 0) && (mp > 0)) {
      if(nq == 1) { // column vector
        int id;
        CHECK_ROCBLAS_ERROR(rocblas_idamax(handle, mp, A, 1, &id));
        CHECK_HIP_ERROR(hipMemcpy(
            &v0, A + id - 1, 1 * sizeof(double), hipMemcpyDeviceToHost));
      } else if(mp == 1) { // row vector
        int id;
        CHECK_ROCBLAS_ERROR(rocblas_idamax(handle, nq, A, LDA, &id));
        CHECK_HIP_ERROR(hipMemcpy(&v0,
                                  A + ((size_t)id * LDA),
                                  1 * sizeof(double),
                                  hipMemcpyDeviceToHost));
      } else {
        // custom reduction kernels
        CHECK_HIP_ERROR(hipMalloc(&dwork, GRID_SIZE * sizeof(double)));

        size_t grid_size = (nq * mp + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_size        = (grid_size < GRID_SIZE) ? grid_size : GRID_SIZE;

        normA_1<<<grid_size, BLOCK_SIZE>>>(nq, mp, A, LDA, dwork);
        CHECK_HIP_ERROR(hipGetLastError());
        normA_2<<<1, BLOCK_SIZE>>>(grid_size, dwork);
        CHECK_HIP_ERROR(hipGetLastError());

        CHECK_HIP_ERROR(
            hipMemcpy(&v0, dwork, 1 * sizeof(double), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipFree(dwork));
      }
    }
    (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_MAX, 0, Acomm);
  } else if(NORM == HPL_NORM_1) {
    /*
     * Find norm_1( A ).
     */
    if(nq > 0) {
      work = (double*)malloc((size_t)(nq) * sizeof(double));
      if(work == NULL) {
        HPL_pabort(__LINE__, "HPL_pdlange", "Memory allocation failed");
      }

      if(nq == 1) { // column vector
        CHECK_ROCBLAS_ERROR(rocblas_dasum(handle, mp, A, 1, work));
      } else {
        CHECK_HIP_ERROR(hipMalloc(&dwork, nq * sizeof(double)));
        norm1<<<nq, BLOCK_SIZE>>>(nq, mp, A, LDA, dwork);
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(
            hipMemcpy(work, dwork, nq * sizeof(double), hipMemcpyDeviceToHost));
      }
      /*
       * Find sum of global matrix columns, store on row 0 of process grid
       */
      (void)HPL_reduce((void*)(work), nq, HPL_DOUBLE, HPL_SUM, 0, Ccomm);
      /*
       * Find maximum sum of columns for 1-norm
       */
      if(myrow == 0) {
        v0 = work[HPL_idamax(nq, work, 1)];
        v0 = Mabs(v0);
      }
      if(work) free(work);
      if(dwork) CHECK_HIP_ERROR(hipFree(dwork));
    }
    /*
     * Find max in row 0, store result in process (0,0)
     */
    if(myrow == 0)
      (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_MAX, 0, Rcomm);
  } else if(NORM == HPL_NORM_I) {
    /*
     * Find norm_inf( A )
     */
    if(mp > 0) {
      work = (double*)malloc((size_t)(mp) * sizeof(double));
      if(work == NULL) {
        HPL_pabort(__LINE__, "HPL_pdlange", "Memory allocation failed");
      }

      if(mp == 1) { // row vector
        CHECK_ROCBLAS_ERROR(rocblas_dasum(handle, nq, A, LDA, work));
      } else {
        CHECK_HIP_ERROR(hipMalloc(&dwork, mp * sizeof(double)));

        size_t grid_size = (mp + BLOCK_SIZE - 1) / BLOCK_SIZE;
        norminf<<<grid_size, BLOCK_SIZE>>>(nq, mp, A, LDA, dwork);
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(
            hipMemcpy(work, dwork, mp * sizeof(double), hipMemcpyDeviceToHost));
      }

      /*
       * Find sum of global matrix rows, store on column 0 of process grid
       */
      (void)HPL_reduce((void*)(work), mp, HPL_DOUBLE, HPL_SUM, 0, Rcomm);
      /*
       * Find maximum sum of rows for inf-norm
       */
      if(mycol == 0) {
        v0 = work[HPL_idamax(mp, work, 1)];
        v0 = Mabs(v0);
      }
      if(work) free(work);
      if(dwork) CHECK_HIP_ERROR(hipFree(dwork));
    }
    /*
     * Find max in column 0, store result in process (0,0)
     */
    if(mycol == 0)
      (void)HPL_reduce((void*)(&v0), 1, HPL_DOUBLE, HPL_MAX, 0, Ccomm);
  }
  /*
   * Broadcast answer to every process in the grid
   */
  (void)HPL_broadcast((void*)(&v0), 1, HPL_DOUBLE, 0, Acomm);

  return (v0);
}
