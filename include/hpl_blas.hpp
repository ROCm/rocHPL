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
#ifndef HPL_BLAS_HPP
#define HPL_BLAS_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */

#include "hpl_misc.hpp"
#include <rocblas/rocblas.h>
#include <roctracer.h>
#include <roctx.h>
#include <iostream>

extern rocblas_handle handle;
extern hipStream_t    computeStream;
extern hipStream_t    dataStream;

#define CHECK_HIP_ERROR(val) hipCheck((val), #val, __FILE__, __LINE__)
inline void hipCheck(hipError_t        err,
                     const char* const func,
                     const char* const file,
                     const int         line) {
  if(err != hipSuccess) {
    std::cerr << "HIP Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << hipGetErrorString(err) << " " << func << std::endl;
    std::exit(-1);
  }
}

#define CHECK_ROCBLAS_ERROR(val) rocBLASCheck((val), #val, __FILE__, __LINE__)
inline void rocBLASCheck(rocblas_status    err,
                         const char* const func,
                         const char* const file,
                         const int         line) {
  if(err != rocblas_status_success) {
    std::cerr << "rocBLAS Reports Error at: " << file << ":" << line
              << std::endl;
    std::cerr << rocblas_status_to_string(err) << " " << func << std::endl;
    std::exit(-1);
  }
}

#if __cplusplus
extern "C" {
#endif

/*
 * ---------------------------------------------------------------------
 * typedef definitions
 * ---------------------------------------------------------------------
 */
enum HPL_ORDER { HplRowMajor = 101, HplColumnMajor = 102 };
enum HPL_TRANS { HplNoTrans = 111, HplTrans = 112, HplConjTrans = 113 };
enum HPL_UPLO { HplUpper = 121, HplLower = 122 };
enum HPL_DIAG { HplNonUnit = 131, HplUnit = 132 };
enum HPL_SIDE { HplLeft = 141, HplRight = 142 };

/*
 * ---------------------------------------------------------------------
 * Blocked OpenMP routines
 * ---------------------------------------------------------------------
 */

void HPL_idamax_omp(const int     N,
                    const double* X,
                    const int     INCX,
                    const int     NB,
                    const int     II,
                    const int     thread_rank,
                    const int     thread_size,
                    int*          max_index,
                    double*       max_value);

void HPL_dscal_omp(const int    N,
                   const double ALPHA,
                   double*      X,
                   const int    INCX,
                   const int    NB,
                   const int    II,
                   const int    thread_rank,
                   const int    thread_size);

void HPL_daxpy_omp(const int     N,
                   const double  ALPHA,
                   const double* X,
                   const int     INCX,
                   double*       Y,
                   const int     INCY,
                   const int     NB,
                   const int     II,
                   const int     thread_rank,
                   const int     thread_size);

void HPL_dger_omp(const enum HPL_ORDER ORDER,
                  const int            M,
                  const int            N,
                  const double         ALPHA,
                  const double*        X,
                  const int            INCX,
                  double*              Y,
                  const int            INCY,
                  double*              A,
                  const int            LDA,
                  const int            NB,
                  const int            II,
                  const int            thread_rank,
                  const int            thread_size);

void HPL_dgemv_omp(const enum HPL_ORDER ORDER,
                   const enum HPL_TRANS TRANS,
                   const int            M,
                   const int            N,
                   const double         ALPHA,
                   const double*        A,
                   const int            LDA,
                   const double*        X,
                   const int            INCX,
                   const double         BETA,
                   double*              Y,
                   const int            INCY,
                   const int            NB,
                   const int            II,
                   const int            thread_rank,
                   const int            thread_size);

void HPL_dgemm_omp(const enum HPL_ORDER ORDER,
                   const enum HPL_TRANS TRANSA,
                   const enum HPL_TRANS TRANSB,
                   const int            M,
                   const int            N,
                   const int            K,
                   const double         ALPHA,
                   const double*        A,
                   const int            LDA,
                   const double*        B,
                   const int            LDB,
                   const double         BETA,
                   double*              C,
                   const int            LDC,
                   const int            NB,
                   const int            II,
                   const int            thread_rank,
                   const int            thread_size);

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define CBLAS_INDEX int

#define CBLAS_ORDER HPL_ORDER
#define CblasRowMajor HplRowMajor
#define CblasColMajor HplColMajor

#define CBLAS_TRANSPOSE HPL_TRANS
#define CblasNoTrans HplNoTrans
#define CblasTrans HplTrans
#define CblasConjTrans HplConjTrans

#define CBLAS_UPLO HPL_UPLO
#define CblasUpper HplUpper
#define CblasLower HplLower

#define CBLAS_DIAG HPL_DIAG
#define CblasNonUnit HplNonUnit
#define CblasUnit HplUnit

#define CBLAS_SIDE HPL_SIDE
#define CblasLeft HplLeft
#define CblasRight HplRight
/*
 * ---------------------------------------------------------------------
 * CBLAS Function prototypes
 * ---------------------------------------------------------------------
 */
CBLAS_INDEX cblas_idamax(const int, const double*, const int);
void        cblas_dswap(const int, double*, const int, double*, const int);
void cblas_dcopy(const int, const double*, const int, double*, const int);

void cblas_daxpy(const int,
                 const double,
                 const double*,
                 const int,
                 double*,
                 const int);

void cblas_dscal(const int, const double, double*, const int);

void cblas_dgemv(const enum CBLAS_ORDER,
                 const enum CBLAS_TRANSPOSE,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 const double*,
                 const int,
                 const double,
                 double*,
                 const int);

void cblas_dger(const enum CBLAS_ORDER,
                const int,
                const int,
                const double,
                const double*,
                const int,
                const double*,
                const int,
                double*,
                const int);

void cblas_dtrsv(const enum CBLAS_ORDER,
                 const enum CBLAS_UPLO,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_DIAG,
                 const int,
                 const double*,
                 const int,
                 double*,
                 const int);

void cblas_dgemm(const enum CBLAS_ORDER,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_TRANSPOSE,
                 const int,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 const double*,
                 const int,
                 const double,
                 double*,
                 const int);

void cblas_dtrsm(const enum CBLAS_ORDER,
                 const enum CBLAS_SIDE,
                 const enum CBLAS_UPLO,
                 const enum CBLAS_TRANSPOSE,
                 const enum CBLAS_DIAG,
                 const int,
                 const int,
                 const double,
                 const double*,
                 const int,
                 double*,
                 const int);
/*
 * ---------------------------------------------------------------------
 * HPL C BLAS macro definition
 * ---------------------------------------------------------------------
 */
#define HPL_dswap cblas_dswap
#define HPL_dcopy cblas_dcopy
#define HPL_daxpy cblas_daxpy
#define HPL_dscal cblas_dscal
#define HPL_idamax cblas_idamax

#define HPL_dgemv cblas_dgemv
#define HPL_dtrsv cblas_dtrsv
#define HPL_dger cblas_dger

#define HPL_dgemm cblas_dgemm
#define HPL_dtrsm cblas_dtrsm

#if __cplusplus
}
#endif

#endif
/*
 * hpl_blas.hpp
 */
