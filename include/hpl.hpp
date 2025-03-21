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
#ifndef HPL_HPP
#define HPL_HPP
/*
 * ---------------------------------------------------------------------
 * HPL default compile options that can overridden in the cmake
 * ---------------------------------------------------------------------
 */
#ifndef HPL_DETAILED_TIMING /* Do not enable detailed timings */
#define HPL_NO_DETAILED_TIMING
#endif

#undef HPL_USE_COLLECTIVES
// #define HPL_USE_COLLECTIVES

// #undef HPL_MXSWP_USE_COLLECTIVES
#define HPL_MXSWP_USE_COLLECTIVES

/*
Enabling atomics will potentially allow more performance optimization
but will potentailly lead to residual values which vary from run-to-run
*/
#undef HPL_ROCBLAS_ALLOW_ATOMICS
// #define HPL_ROCBLAS_ALLOW_ATOMICS

/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include <omp.h>

// NC: hipcc in ROCm 3.7 complains if __HIP_PLATFORM_HCC__ is defined in the
// compile line
#ifdef __HIPCC__
#ifdef __HIP_PLATFORM_HCC__
#undef __HIP_PLATFORM_HCC__
#endif
#endif
#include "hip/hip_runtime_api.h"

#ifdef HPL_TRACING
#include <roctracer.h>
#include <roctx.h>
#endif

#include "hpl_version.hpp"
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pfact.hpp"
#include "hpl_pgesv.hpp"

#include "hpl_ptimer.hpp"
#include "hpl_pmatgen.hpp"
#include "hpl_ptest.hpp"

#endif
/*
 * End of hpl.hpp
 */
