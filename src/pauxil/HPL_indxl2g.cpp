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

int HPL_indxl2g(const int IL,
                const int INB,
                const int NB,
                const int PROC,
                const int SRCPROC,
                const int NPROCS) {
  /*
   * Purpose
   * =======
   *
   * HPL_indxl2g computes the global index of a matrix  entry  pointed to
   * by the local index IL of the process indicated by PROC.
   *
   * Arguments
   * =========
   *
   * IL      (input)                       const int
   *         On entry, IL specifies the local  index of the matrix  entry.
   *         IL must be at least zero.
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
   *         On entry, PROC  specifies the coordinate of the process whose
   *         local array row or column is to be determined. PROC  must  be
   *         at least zero and strictly less than NPROCS.
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

  if((SRCPROC == -1) || (NPROCS == 1)) {
    /*
     * The data is not distributed, or there is just one process in this di-
     * mension of the grid.
     */
    return (IL);
  } else if(PROC == SRCPROC) {
    /*
     * If I am SRCPROC, my first block is of size INB
     */
    if(IL < INB)
      /*
       * If  IL  belongs to the first block,  the local and global indexes are
       * equal.
       */
      return (IL);
    /*
     * The  number  of  entire  blocks  before  the  one  IL  belongs  to is
     * ( IL - INB ) / NB + 1.  In  the other NPROCS-1 processes,  there  are
     * thus NB*( ( IL-INB )/NB + 1 ) entries,  that are  globally before the
     * global entry corresponding to IL.
     */
    return ((NPROCS - 1) * NB * ((IL - INB) / NB + 1) + IL);
  } else if(PROC < SRCPROC) {
    /*
     * Otherwise, the process of coordinate  MOD(SRCPROC+1, NPROCS) owns the
     * second block. Let IPROC = PROC-SRCPROC-1+NPROCS be the number of pro-
     * cesses between this process and  PROC  not  included  when going from
     * left to right on the process line  with  possible wrap around.  These
     * IPROC  processes have one more NB block than the other processes, who
     * own IL / NB blocks of size NB.
     */
    return (NB * ((NPROCS - 1) * (IL / NB) + PROC - SRCPROC - 1 + NPROCS) + IL +
            INB);
  } else {
    /*
     * Same reasoning as above with IPROC = PROC - SRCPROC - 1.
     */
    return (NB * ((NPROCS - 1) * (IL / NB) + PROC - SRCPROC - 1) + IL + INB);
  }
}
