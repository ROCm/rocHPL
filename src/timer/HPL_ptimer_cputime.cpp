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

/*
 * Purpose
 * =======
 *
 * HPL_ptimer_cputime returns the cpu time.
 * The  clock() function is used to return an approximation of processor
 * time used by the program.  The value returned is the CPU time used so
 * far as a clock_t;  to get the number of seconds used,  the result  is
 * divided by  CLOCKS_PER_SEC.  This function is part of the  ANSI/ISO C
 * standard library.
 *
 * ---------------------------------------------------------------------
 */

#include <time.h>

double HPL_ptimer_cputime(void) {
  static double  cps = CLOCKS_PER_SEC;
  double         d;
  clock_t        t1;
  static clock_t t0 = 0;

  if(t0 == 0) t0 = clock();
  t1 = clock() - t0;
  d  = (double)(t1) / cps;
  return (d);
}
