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
#ifndef HPL_BLAS_HPP
#define HPL_BLAS_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */

#include "hpl_misc.hpp"
#include <rocblas/rocblas.h>
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

#endif
/*
 * hpl_blas.hpp
 */
