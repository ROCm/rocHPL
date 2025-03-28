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
#ifndef HPL_PTIMER_HPP
#define HPL_PTIMER_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.hpp"
#include <chrono>

using timePoint_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_NPTIMER 64
#define HPL_PTIMER_STARTFLAG 5.0
#define HPL_PTIMER_ERROR -1.0
/*
 * ---------------------------------------------------------------------
 * type definitions
 * ---------------------------------------------------------------------
 */
typedef enum { HPL_WALL_PTIME = 101, HPL_CPU_PTIME = 102 } HPL_T_PTIME;

typedef enum {
  HPL_AMAX_PTIME = 201,
  HPL_AMIN_PTIME = 202,
  HPL_SUM_PTIME  = 203
} HPL_T_PTIME_OP;
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */
double HPL_ptimer_cputime(void);
double HPL_ptimer_walltime(void);
void   HPL_ptimer(const int);
void   HPL_ptimer_boot(void);

void HPL_ptimer_combine(MPI_Comm comm,
                        const HPL_T_PTIME_OP,
                        const HPL_T_PTIME,
                        const int,
                        const int,
                        double*);

void   HPL_ptimer_disable(void);
void   HPL_ptimer_enable(void);
double HPL_ptimer_inquire(const HPL_T_PTIME, const int);
void   HPL_ptimer_stepReset(const int, const int);
double HPL_ptimer_getStep(const int);

#endif
/*
 * End of hpl_ptimer.hpp
 */
