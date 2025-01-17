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
#ifndef HPL_PTEST_HPP
#define HPL_PTEST_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pgesv.hpp"

#include "hpl_ptimer.hpp"
#include "hpl_pmatgen.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
typedef struct HPL_S_test {
  double epsil; /* epsilon machine */
  double thrsh; /* threshold */
  FILE*  outfp; /* output stream (only in proc 0) */
  int    kfail; /* # of tests failed */
  int    kpass; /* # of tests passed */
  int    kskip; /* # of tests skipped */
  int    ktest; /* total number of tests */
} HPL_T_test;

/*
 * ---------------------------------------------------------------------
 * #define macro constants for testing only
 * ---------------------------------------------------------------------
 */
#define HPL_LINE_MAX 256
#define HPL_MAX_PARAM 20
#define HPL_ISEED 100
/*
 * ---------------------------------------------------------------------
 * global timers for timing analysis only
 * ---------------------------------------------------------------------
 */
#define HPL_TIMING_BEG 11    /* timer 0 reserved, used by main */
#define HPL_TIMING_N 8       /* number of timers defined below */
#define HPL_TIMING_RPFACT 11 /* starting from here, contiguous */
#define HPL_TIMING_PFACT 12
#define HPL_TIMING_MXSWP 13
#define HPL_TIMING_COPY 14
#define HPL_TIMING_LBCAST 15
#define HPL_TIMING_LASWP 16
#define HPL_TIMING_UPDATE 17
#define HPL_TIMING_PTRSV 18
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
void HPL_pdinfo(int    ARGC,
                char** ARGV,
                HPL_T_test*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_ORDER*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_FACT*,
                int*,
                int*,
                int*,
                int*,
                int*,
                HPL_T_FACT*,
                int*,
                HPL_T_TOP*,
                int*,
                int*,
                HPL_T_SWAP*,
                int*,
                int*,
                int*,
                int*,
                int*,
                double*);

int HPL_pdwarmup(HPL_T_test* TEST,
                 HPL_T_grid* GRID,
                 HPL_T_palg* ALGO,
                 HPL_T_pmat* mat);

void HPL_pdtest(HPL_T_test*, HPL_T_grid*, HPL_T_palg*, const int, const int);
void HPL_InitGPU(const HPL_T_grid* GRID);
void HPL_FreeGPU();

#endif
/*
 * End of hpl_ptest.hpp
 */
