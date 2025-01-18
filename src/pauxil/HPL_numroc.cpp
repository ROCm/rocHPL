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

#include "hpl.hpp"

int HPL_numroc(const int N,
               const int INB,
               const int NB,
               const int PROC,
               const int SRCPROC,
               const int NPROCS) {
  /*
   * Purpose
   * =======
   *
   * HPL_numroc returns  the  local number of matrix rows/columns process
   * PROC  will  get  if  we give out  N rows/columns starting from global
   * index 0.
   *
   * Arguments
   * =========
   *
   * N       (input)                       const int
   *         On entry, N  specifies the number of rows/columns being dealt
   *         out. N must be at least zero.
   *
   * INB     (input)                       const int
   *         On entry,  INB  specifies  the size of the first block of the
   *         global matrix. INB must be at least one.
   *
   * NB      (input)                       const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix A. NB must be larger than one.
   *
   * PROC    (input)                       const int
   *         On entry, PROC specifies  the coordinate of the process whose
   *         local portion is determined.  PROC must be at least zero  and
   *         strictly less than NPROCS.
   *
   * SRCPROC (input)                       const int
   *         On entry,  SRCPROC  specifies  the coordinate of the  process
   *         that possesses the first row or column of the matrix. SRCPROC
   *         must be at least zero and strictly less than NPROCS.
   *
   * NPROCS  (input)                       const int
   *         On entry,  NPROCS  specifies the total number of process rows
   *         or columns over which the matrix is distributed.  NPROCS must
   *         be at least one.
   *
   * ---------------------------------------------------------------------
   */

  return (HPL_numrocI(N, 0, INB, NB, PROC, SRCPROC, NPROCS));
}
