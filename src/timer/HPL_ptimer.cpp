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
 * ---------------------------------------------------------------------
 * Static variables
 * ---------------------------------------------------------------------
 */
static int    HPL_ptimer_disabled;
static double HPL_ptimer_cpusec[HPL_NPTIMER], HPL_ptimer_cpustart[HPL_NPTIMER];
static double HPL_ptimer_wallsec[HPL_NPTIMER],
    HPL_ptimer_wallstart[HPL_NPTIMER];
static double HPL_ptimer_wallstep[HPL_NPTIMER];
/*
 * ---------------------------------------------------------------------
 * User callable functions
 * ---------------------------------------------------------------------
 */
void HPL_ptimer_boot() {
  /*
   * HPL_ptimer_boot (re)sets all timers to 0, and enables HPL_ptimer.
   */

  int i;

  HPL_ptimer_disabled = 0;

  for(i = 0; i < HPL_NPTIMER; i++) {
    HPL_ptimer_cpusec[i] = HPL_ptimer_wallsec[i] = HPL_rzero;
    HPL_ptimer_wallstep[i]                       = HPL_rzero;
    HPL_ptimer_cpustart[i] = HPL_ptimer_wallstart[i] = HPL_PTIMER_STARTFLAG;
  }
}

void HPL_ptimer(const int I) {
  /*
   * Purpose
   * =======
   *
   * HPL_ptimer provides a  "stopwatch"  functionality  cpu/wall  timer in
   * seconds.  Up to  64  separate timers can be functioning at once.  The
   * first call starts the timer,  and the second stops it.  This  routine
   * can be disenabled  by calling HPL_ptimer_disable(),  so that calls to
   * the timer are ignored.  This feature can be used to make sure certain
   * sections of code do not affect timings,  even  if  they call routines
   * which have HPL_ptimer calls in them. HPL_ptimer_enable()  will enable
   * the  timer  functionality.  One  can retrieve  the current value of a
   * timer by calling
   *
   * t0 = HPL_ptimer_inquire( HPL_WALL_TIME | HPL_CPU_TIME, I )
   *
   * where  I  is the timer index in  [0..64).  To  inititialize the timer
   * functionality, one must have called HPL_ptimer_boot() prior to any of
   * the functions mentioned above.
   *
   * Arguments
   * =========
   *
   * I       (global input)                const int
   *         On entry, I specifies the timer to stop/start.
   *
   * ---------------------------------------------------------------------
   */

  if(HPL_ptimer_disabled) return;
  /*
   * If timer has not been started, start it.  Otherwise,  stop it and add
   * interval to count
   */
  if(HPL_ptimer_wallstart[I] == HPL_PTIMER_STARTFLAG) {
    HPL_ptimer_wallstart[I] = HPL_ptimer_walltime();
    HPL_ptimer_cpustart[I]  = HPL_ptimer_cputime();
  } else {
    HPL_ptimer_cpusec[I] += HPL_ptimer_cputime() - HPL_ptimer_cpustart[I];
    const double walltime = HPL_ptimer_walltime() - HPL_ptimer_wallstart[I];
    HPL_ptimer_wallstep[I] += walltime;
    HPL_ptimer_wallsec[I] += walltime;
    HPL_ptimer_wallstart[I] = HPL_PTIMER_STARTFLAG;
  }
}

void HPL_ptimer_enable(void) {
  /*
   * HPL_ptimer_enable sets it so calls to HPL_ptimer are not ignored.
   */

  HPL_ptimer_disabled = 0;
  return;
}

void HPL_ptimer_disable(void) {
  /*
   * HPL_ptimer_disable sets it so calls to HPL_ptimer are ignored.
   */

  HPL_ptimer_disabled = 1;
  return;
}

void HPL_ptimer_stepReset(const int N, const int IBEG) {
  for(int i = 0; i < N; i++) { HPL_ptimer_wallstep[IBEG + i] = HPL_rzero; }
}

double HPL_ptimer_getStep(const int I) {

  double time;

  /*
   * If wall-time are not available on this machine, return
   * HPL_PTIMER_ERROR
   */
  if(HPL_ptimer_walltime() == HPL_PTIMER_ERROR)
    time = HPL_PTIMER_ERROR;
  else
    time = HPL_ptimer_wallstep[I];

  return (time);
}

double HPL_ptimer_inquire(const HPL_T_PTIME TMTYPE, const int I) {
  /*
   * Purpose
   * =======
   *
   * HPL_ptimer_inquire returns wall- or cpu- time that has accumulated in
   * timer I.
   *
   * Arguments
   * =========
   *
   * TMTYPE  (global input)              const HPL_T_PTIME
   *         On entry, TMTYPE specifies what time will be returned as fol-
   *         lows
   *            = HPL_WALL_PTIME : wall clock time is returned,
   *            = HPL_CPU_PTIME  : CPU time is returned (default).
   *
   * I       (global input)              const int
   *         On entry, I specifies the timer to return.
   *
   * ---------------------------------------------------------------------
   */

  double time;

  /*
   * If wall- or cpu-time are not available on this machine, return
   * HPL_PTIMER_ERROR
   */
  if(TMTYPE == HPL_WALL_PTIME) {
    if(HPL_ptimer_walltime() == HPL_PTIMER_ERROR)
      time = HPL_PTIMER_ERROR;
    else
      time = HPL_ptimer_wallsec[I];
  } else {
    if(HPL_ptimer_cputime() == HPL_PTIMER_ERROR)
      time = HPL_PTIMER_ERROR;
    else
      time = HPL_ptimer_cpusec[I];
  }
  return (time);
}

void HPL_ptimer_combine(MPI_Comm             COMM,
                        const HPL_T_PTIME_OP OPE,
                        const HPL_T_PTIME    TMTYPE,
                        const int            N,
                        const int            IBEG,
                        double*              TIMES) {
  /*
   * Purpose
   * =======
   *
   * HPL_ptimer_combine  combines the timing information stored on a scope
   * of processes into the user TIMES array.
   *
   * Arguments
   * =========
   *
   * COMM    (global/local input)        MPI_Comm
   *         The MPI communicator  identifying  the process  collection on
   *         which the timings are taken.
   *
   * OPE     (global input)              const HPL_T_PTIME_OP
   *         On entry, OP  specifies what combine operation should be done
   *         as follows:
   *            = HPL_AMAX_PTIME get max. time on any process (default),
   *            = HPL_AMIN_PTIME get min. time on any process,
   *            = HPL_SUM_PTIME  get sum of times across processes.
   *
   * TMTYPE  (global input)              const HPL_T_PTIME
   *         On entry, TMTYPE specifies what time will be returned as fol-
   *         lows
   *            = HPL_WALL_PTIME : wall clock time is returned,
   *            = HPL_CPU_PTIME  : CPU time is returned (default).
   *
   * N       (global input)              const int
   *         On entry, N specifies the number of timers to combine.
   *
   * IBEG    (global input)              const int
   *         On entry, IBEG specifies the first timer to be combined.
   *
   * TIMES   (global output)             double *
   *         On entry, TIMES is an array of dimension at least N. On exit,
   *         this array contains the requested timing information.
   *
   * ---------------------------------------------------------------------
   */

  int i, tmpdis;

  tmpdis              = HPL_ptimer_disabled;
  HPL_ptimer_disabled = 1;
  /*
   * Timer has been disabled for combine operation -  copy timing informa-
   * tion into user times array.  If  wall- or  cpu-time are not available
   * on this machine, fill in times with HPL_PTIMER_ERROR flag and return.
   */
  if(TMTYPE == HPL_WALL_PTIME) {
    if(HPL_ptimer_walltime() == HPL_PTIMER_ERROR) {
      for(i = 0; i < N; i++) TIMES[i] = HPL_PTIMER_ERROR;
      return;
    } else {
      for(i = 0; i < N; i++) TIMES[i] = HPL_ptimer_wallsec[IBEG + i];
    }
  } else {
    if(HPL_ptimer_cputime() == HPL_PTIMER_ERROR) {
      for(i = 0; i < N; i++) TIMES[i] = HPL_PTIMER_ERROR;
      return;
    } else {
      for(i = 0; i < N; i++) TIMES[i] = HPL_ptimer_cpusec[IBEG + i];
    }
  }
  /*
   * Combine all nodes information, restore HPL_ptimer_disabled, and return
   */
  for(i = 0; i < N; i++) TIMES[i] = Mmax(HPL_rzero, TIMES[i]);

  if(OPE == HPL_AMAX_PTIME)
    (void)HPL_all_reduce((void*)(TIMES), N, HPL_DOUBLE, HPL_MAX, COMM);
  else if(OPE == HPL_AMIN_PTIME)
    (void)HPL_all_reduce((void*)(TIMES), N, HPL_DOUBLE, HPL_MIN, COMM);
  else if(OPE == HPL_SUM_PTIME)
    (void)HPL_all_reduce((void*)(TIMES), N, HPL_DOUBLE, HPL_SUM, COMM);
  else
    (void)HPL_all_reduce((void*)(TIMES), N, HPL_DOUBLE, HPL_MAX, COMM);

  HPL_ptimer_disabled = tmpdis;
}
