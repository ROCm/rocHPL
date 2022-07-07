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

void HPL_pabort(int LINE, const char* SRNAME, const char* FORM, ...) {
  /*
   * Purpose
   * =======
   *
   * HPL_pabort displays an error message on stderr and halts execution.
   *
   *
   * Arguments
   * =========
   *
   * LINE    (local input)                 int
   *         On entry,  LINE  specifies the line  number in the file where
   *         the  error  has  occured.  When  LINE  is not a positive line
   *         number, it is ignored.
   *
   * SRNAME  (local input)                 const char *
   *         On entry, SRNAME  should  be the name of the routine  calling
   *         this error handler.
   *
   * FORM    (local input)                 const char *
   *         On entry, FORM specifies the format, i.e., how the subsequent
   *         arguments are converted for output.
   *
   *         (local input)                 ...
   *         On entry,  ...  is the list of arguments to be printed within
   *         the format string.
   *
   * ---------------------------------------------------------------------
   */

  va_list argptr;
  int     rank;
  char    cline[128];

  va_start(argptr, FORM);
  (void)vsprintf(cline, FORM, argptr);
  va_end(argptr);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /*
   * Display an error message
   */
  if(LINE <= 0)
    HPL_fprintf(stderr,
                "%s %s %d, %s %s:\n>>> %s <<< Abort ...\n\n",
                "HPL ERROR",
                "from process #",
                rank,
                "in function",
                SRNAME,
                cline);
  else
    HPL_fprintf(stderr,
                "%s %s %d, %s %d %s %s:\n>>> %s <<< Abort ...\n\n",
                "HPL ERROR",
                "from process #",
                rank,
                "on line",
                LINE,
                "of function",
                SRNAME,
                cline);

  MPI_Abort(MPI_COMM_WORLD, -1);
  exit(-1);
}
