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

void HPL_indxg2lp(int*      IL,
                  int*      PROC,
                  const int IG,
                  const int INB,
                  const int NB,
                  const int SRCPROC,
                  const int NPROCS) {
  /*
   * Purpose
   * =======
   *
   * HPL_indxg2lp computes the local index of a matrix entry pointed to by
   * the global  index IG as well as the process coordinate which posseses
   * this entry. The local returned index is the same in all processes.
   *
   * Arguments
   * =========
   *
   * IL      (output)                      int *
   *         On exit, IL specifies the local index corresponding to IG. IL
   *         is at least zero.
   *
   * PROC    (output)                      int *
   *         On exit,  PROC  is the  coordinate of the process  owning the
   *         entry specified by the global index IG. PROC is at least zero
   *         and less than NPROCS.
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
   *         On entry, if SRCPROC = -1, the data  is not  distributed  but
   *         replicated,  in  which  case  this  routine returns IG in all
   *         processes. Otherwise, the value of SRCPROC is ignored.
   *
   * NPROCS  (input)                       const int
   *         On entry,  NPROCS  specifies the total number of process rows
   *         or columns over which the matrix is distributed.  NPROCS must
   *         be at least one.
   *
   * ---------------------------------------------------------------------
   */

  int i, j;

  if((IG < INB) || (SRCPROC == -1) || (NPROCS == 1)) {
    /*
     * IG  belongs  to the first block,  or the data is not distributed,  or
     * there is just one process in this dimension of the grid.
     */
    *IL   = IG;
    *PROC = SRCPROC;
  } else {
    /*
     * IG  =  INB - NB + ( l * NPROCS + MYROC ) * NB + X  with  0 <= X < NB,
     * thus IG is to be found in the block (IG-INB+NB) / NB = l*NPROCS+MYROC
     * with  0 <= MYROC < NPROCS.  The local index to be returned depends on
     * whether  IG  resides in the process owning the first partial block of
     * size INB (MYROC=0). To determine this cheaply, let i = (IG-INB) / NB,
     * so that if NPROCS divides i+1, i.e. MYROC=0,  we have i+1 = l*NPROCS.
     * If we set  j = i / NPROCS, it follows that j = l-1. Therefore, i+1 is
     * equal to (j+1) * NPROCS.  Conversely, if NPROCS does not divide  i+1,
     * then i+1 = l*NPROCS + MYROC with 1 <= MYROC < NPROCS. It follows that
     * j=l and thus (j+1)*NPROCS > i+1.
     */
    j = (i = (IG - INB) / NB) / NPROCS;
    /*
     * IG  is in block  1 + ( IG - INB ) / NB.  Add this to SRCPROC and take
     * the NPROCS modulo (definition of the block-cyclic data distribution).
     */
    *PROC = SRCPROC + 1 + i;
    *PROC = MPosMod(*PROC, NPROCS);
    /*
     * When IG resides in the process owning the first partial block of size
     * INB (MYROC = 0), then the result IL can be written as:
     * IL = INB - NB + l * NB + X  = IG + ( l - (l * NPROCS + MYROC) ) * NB.
     * Using the above notation,  we have i+1 = l*NPROCS + MYROC = l*NPROCS,
     * i.e l = ( i+1 ) / NPROCS = j+1,  since  NPROCS divides i+1, therefore
     * IL = IG + ( j + 1 - ( i + 1 ) ) * NB.
     *
     * Otherwise when MYROC >= 1, the result IL can be written as:
     * IL = l * NB + X = IG - INB + ( ( l+1 ) - ( l * NPROCS + MYROC ) )*NB.
     * We still have i+1 = l*NPROCS+MYROC. Since NPROCS does not divide i+1,
     * we have j = (l*NPROCS+MYROC-1) / NPROCS = l, i.e
     * IL = IG - INB + ( j + 1 - ( i + 1 ) ) * NB.
     */
    *IL = NB * (j - i) + ((i + 1 - (j + 1) * NPROCS) ? IG - INB : IG);
  }
}
