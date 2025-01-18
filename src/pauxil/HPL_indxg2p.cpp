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

int HPL_indxg2p(const int IG,
                const int INB,
                const int NB,
                const int SRCPROC,
                const int NPROCS) {
  /*
   * Purpose
   * =======
   *
   * HPL_indxg2p computes the process coordinate  which posseses the entry
   * of a matrix specified by a global index IG.
   *
   * Arguments
   * =========
   *
   * IG      (input)                       const int
   *         On entry, IG specifies the global index of the matrix  entry.
   *         IG must be at least zero.
   *
   * INB     (input)                       const int
   *         On entry,  INB  specifies  the size of the first block of the
   *         global matrix. INB must be at least one.
   *
   * NB      (input)                       const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix A. NB must be larger than one.
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

  int proc;

  if((IG < INB) || (SRCPROC == -1) || (NPROCS == 1))
    /*
     * IG  belongs  to the first block,  or the data is not distributed,  or
     * there is just one process in this dimension of the grid.
     */
    return (SRCPROC);
  /*
   * Otherwise,  IG is in block 1 + ( IG - INB ) / NB. Add this to SRCPROC
   * and take the NPROCS  modulo (definition of the block-cyclic data dis-
   * tribution).
   */
  proc = SRCPROC + 1 + (IG - INB) / NB;
  return (MPosMod(proc, NPROCS));
}
