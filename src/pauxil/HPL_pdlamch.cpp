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

double HPL_pdlamch(MPI_Comm COMM, const HPL_T_MACH CMACH) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdlamch determines  machine-specific  arithmetic  constants  such  as
   * the relative machine precision (eps),  the safe minimum(sfmin) such that
   * 1/sfmin does not overflow, the base of the machine (base), the precision
   * (prec),  the  number  of  (base)  digits in the  mantissa  (t),  whether
   * rounding occurs in addition (rnd = 1.0 and 0.0 otherwise),  the  minimum
   * exponent before  (gradual)  underflow (emin),  the  underflow  threshold
   * (rmin)- base**(emin-1), the largest exponent before overflow (emax), the
   * overflow threshold (rmax)  - (base**emax)*(1-eps).
   *
   * Arguments
   * =========
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * CMACH   (global input)                const HPL_T_MACH
   *         Specifies the value to be returned by HPL_pdlamch
   *            = HPL_MACH_EPS,   HPL_pdlamch := eps (default)
   *            = HPL_MACH_SFMIN, HPL_pdlamch := sfmin
   *            = HPL_MACH_BASE,  HPL_pdlamch := base
   *            = HPL_MACH_PREC,  HPL_pdlamch := eps*base
   *            = HPL_MACH_MLEN,  HPL_pdlamch := t
   *            = HPL_MACH_RND,   HPL_pdlamch := rnd
   *            = HPL_MACH_EMIN,  HPL_pdlamch := emin
   *            = HPL_MACH_RMIN,  HPL_pdlamch := rmin
   *            = HPL_MACH_EMAX,  HPL_pdlamch := emax
   *            = HPL_MACH_RMAX,  HPL_pdlamch := rmax
   *
   *         where
   *
   *            eps   = relative machine precision,
   *            sfmin = safe minimum,
   *            base  = base of the machine,
   *            prec  = eps*base,
   *            t     = number of digits in the mantissa,
   *            rnd   = 1.0 if rounding occurs in addition,
   *            emin  = minimum exponent before underflow,
   *            rmin  = underflow threshold,
   *            emax  = largest exponent before overflow,
   *            rmax  = overflow threshold.
   *
   * ---------------------------------------------------------------------
   */

  double param;

  param = HPL_dlamch(CMACH);

  switch(CMACH) {
    case HPL_MACH_EPS:
    case HPL_MACH_SFMIN:
    case HPL_MACH_EMIN:
    case HPL_MACH_RMIN:
      (void)HPL_all_reduce((void*)(&param), 1, HPL_DOUBLE, HPL_MAX, COMM);
      break;
    case HPL_MACH_EMAX:
    case HPL_MACH_RMAX:
      (void)HPL_all_reduce((void*)(&param), 1, HPL_DOUBLE, HPL_MIN, COMM);
      break;
    default: break;
  }

  return (param);
}
