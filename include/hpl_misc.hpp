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
#ifndef HPL_MISC_HPP
#define HPL_MISC_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_rone 1.0
#define HPL_rtwo 2.0
#define HPL_rzero 0.0
/*
 * ---------------------------------------------------------------------
 * #define macros definitions
 * ---------------------------------------------------------------------
 */
#define Mabs(a_) (((a_) < 0) ? -(a_) : (a_))
#define Mmin(a_, b_) (((a_) < (b_)) ? (a_) : (b_))
#define Mmax(a_, b_) (((a_) > (b_)) ? (a_) : (b_))

#define Mfloor(a, b) (((a) > 0) ? (((a) / (b))) : (-(((-(a)) + (b) - 1) / (b))))
#define Mceil(a, b) (((a) + (b) - 1) / (b))
#define Miceil(a, b) (((a) > 0) ? ((((a) + (b) - 1) / (b))) : (-((-(a)) / (b))))

#define Mupcase(C) (((C) > 96 && (C) < 123) ? (C) & 0xDF : (C))
#define Mlowcase(C) (((C) > 64 && (C) < 91) ? (C) | 32 : (C))
/*
 * Mptr returns a pointer to a_( i_, j_ ) for readability reasons and
 * also less silly errors ...
 */
#define Mptr(a_, i_, j_, lda_) \
  ((a_) + (size_t)(i_) + (size_t)(j_) * (size_t)(lda_))
/*
 * Align pointer
 */
#define HPL_PTR(ptr_, al_) ((((size_t)(ptr_) + (al_) - 1) / (al_)) * (al_))
#endif

#ifdef HPL_TRACING
#define HPL_TracingPush(label) roctxRangePush(label)
#define HPL_TracingPop(label) roctxRangePop()
#else
#define HPL_TracingPush(label)
#define HPL_TracingPop(label)
#endif
/*
 * End of hpl_misc.hpp
 */
