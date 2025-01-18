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
#ifndef HPL_AUXIL_HPP
#define HPL_AUXIL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
/*
 * ---------------------------------------------------------------------
 * typedef definitions
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPL_NORM_A = 800,
  HPL_NORM_1 = 801,
  HPL_NORM_I = 802
} HPL_T_NORM;

typedef enum {
  HPL_MACH_EPS   = 900, /* relative machine precision */
  HPL_MACH_SFMIN = 901, /* safe minimum st 1/sfmin does not overflow */
  HPL_MACH_BASE  = 902, /* base = base of the machine */
  HPL_MACH_PREC  = 903, /* prec  = eps*base */
  HPL_MACH_MLEN  = 904, /* number of (base) digits in the mantissa */
  HPL_MACH_RND   = 905, /* 1.0 if rounding occurs in addition */
  HPL_MACH_EMIN  = 906, /* min exponent before (gradual) underflow */
  HPL_MACH_RMIN  = 907, /* underflow threshold base**(emin-1) */
  HPL_MACH_EMAX  = 908, /* largest exponent before overflow */
  HPL_MACH_RMAX  = 909  /* overflow threshold - (base**emax)*(1-eps) */

} HPL_T_MACH;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_fprintf(FILE*, const char*, ...);
void HPL_warn(FILE*, int, const char*, const char*, ...);
void HPL_abort(int, const char*, const char*, ...);

void HPL_dlacpy(const int,
                const int,
                const double*,
                const int,
                double*,
                const int);

void HPL_dlatcpy(const int,
                 const int,
                 const double*,
                 const int,
                 double*,
                 const int);

void HPL_dlacpy_gpu(const int,
                    const int,
                    const double*,
                    const int,
                    double*,
                    const int);

void HPL_dlatcpy_gpu(const int,
                     const int,
                     const double*,
                     const int,
                     double*,
                     const int);

double HPL_dlamch(const HPL_T_MACH);

#endif
/*
 * End of hpl_auxil.hpp
 */
