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

int HPL_numrocI(const int N,
                const int I,
                const int INB,
                const int NB,
                const int PROC,
                const int SRCPROC,
                const int NPROCS) {
  /*
   * Purpose
   * =======
   *
   * HPL_numrocI returns  the  local number of matrix rows/columns process
   * PROC  will  get  if  we give out  N rows/columns starting from global
   * index I.
   *
   * Arguments
   * =========
   *
   * N       (input)                       const int
   *         On entry, N  specifies the number of rows/columns being dealt
   *         out. N must be at least zero.
   *
   * I       (input)                       const int
   *         On entry, I  specifies the global index of the matrix  entry
   *         I must be at least zero.
   *
   * INB     (input)                       const int
   *         On entry,  INB  specifies  the size of the first block of th
   *         global matrix. INB must be at least one.
   *
   * NB      (input)                       const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix A. NB must be larger than one.
   *
   * PROC    (input)                       const int
   *         On entry, PROC specifies  the coordinate of the process whos
   *         local portion is determined.  PROC must be at least zero  an
   *         strictly less than NPROCS.
   *
   * SRCPROC (input)                       const int
   *         On entry,  SRCPROC  specifies  the coordinate of the  proces
   *         that possesses the first row or column of the matrix. SRCPRO
   *         must be at least zero and strictly less than NPROCS.
   *
   * NPROCS  (input)                       const int
   *         On entry,  NPROCS  specifies the total number of process row
   *         or columns over which the matrix is distributed.  NPROCS mus
   *         be at least one.
   *
   * ---------------------------------------------------------------------
   */

  int ilocblk, inb, mydist, nblocks, srcproc;

  if((SRCPROC == -1) || (NPROCS == 1))
    /*
     * The data is not distributed, or there is just one process in this di-
     * mension of the grid.
     */
    return (N);
  /*
   * Compute coordinate of process owning I and corresponding INB
   */
  srcproc = SRCPROC;

  if((inb = INB - I) <= 0) {
    /*
     * I is not in the first block, find out which process has it and update
     * the size of first block
     */
    srcproc += (nblocks = (-inb) / NB + 1);
    srcproc -= (srcproc / NPROCS) * NPROCS;
    inb += nblocks * NB;
  }
  /*
   * Now  everything  is  just like  N, I=0, INB, NB, srcproc, NPROCS. The
   * discussion goes as follows:  compute my distance from the source pro-
   * cess  so that within this process coordinate system,  the source pro-
   * cess is the process such that mydist = 0, or PROC == srcproc.
   *
   * Find  out  how  many  full  blocks are globally (nblocks) and locally
   * (ilocblk) in those N entries. Then remark that
   *
   * when  mydist < nblocks - ilocblk*NPROCS, I own ilocblk+1 full blocks,
   * when  mydist > nblocks - ilocblk*NPROCS, I own ilocblk   full blocks,
   * when  mydist = nblocks - ilocblk*NPROCS, either the last block is not
   * full and I own it,  or the last block is full and I am the first pro-
   * cess owning only ilocblk full blocks.
   */
  if(PROC == srcproc) {
    /*
     * I am the source process, i.e. I own I (mydist=0).  When N <= INB, the
     * answer is simply N.
     */
    if(N <= inb) return (N);
    /*
     * Find  out  how  many  full  blocks are globally (nblocks) and locally
     * (ilocblk) in those N entries.
     */
    nblocks = (N - inb) / NB + 1;
    /*
     * Since  mydist = 0 and nblocks - ilocblk * NPROCS >= 0, there are only
     * two possible cases:
     *
     *   1) When mydist = nblocks - ilocblk * NPROCS = 0, that is NPROCS di-
     *      vides the global number of full blocks,  then the source process
     *      srcproc owns one more block than the other processes;  and N can
     *      be rewritten as N = INB + (nblocks-1) * NB + LNB  with  LNB >= 0
     *      size of the last block. Similarly, the local value Np correspon-
     *      ding to N can be written as  Np = INB + (ilocblk-1) * NB + LNB =
     *      N + ( ilocblk-1 - (nblocks-1) )*NB.  Note  that this case cannot
     *      happen when ilocblk is zero, since nblocks is at least one.
     *
     *   2) mydist = 0 < nblocks - ilocblk * NPROCS, the source process only
     *      owns full blocks,  and  therefore Np = INB + ilocblk * NB.  Note
     *      that when ilocblk is zero, Np is just INB.
     */
    if(nblocks < NPROCS) return (inb);

    ilocblk = nblocks / NPROCS;
    return ((nblocks - ilocblk * NPROCS) ? inb + ilocblk * NB
                                         : N + (ilocblk - nblocks) * NB);
  } else {
    /*
     * I am not the source process. When N <= INB, the answer is simply 0.
     */
    if(N <= inb) return (0);
    /*
     * Find  out  how  many  full  blocks are globally (nblocks) and locally
     * (ilocblk) in those N entries
     */
    nblocks = (N - inb) / NB + 1;
    /*
     * Compute  my distance from the source process so that within this pro-
     * cess coordinate system,  the source  process is the process such that
     * mydist=0.
     */
    if((mydist = PROC - srcproc) < 0) mydist += NPROCS;
    /*
     * When mydist < nblocks - ilocblk*NPROCS, I own ilocblk + 1 full blocks
     * of size NB since I am not the source process,
     *
     * when mydist > nblocks - ilocblk * NPROCS, I own ilocblk   full blocks
     * of size NB since I am not the source process,
     *
     * when mydist = nblocks - ilocblk*NPROCS,
     * either the last block is not full and I own it, in which case
     *    N = INB + (nblocks - 1)*NB + LNB with  LNB  the  size  of the last
     *    block such that NB > LNB > 0;  the local value Np corresponding to
     *    N is given by  Np = ilocblk*NB+LNB = N-INB+(ilocblk-nblocks+1)*NB;
     * or the  last  block  is  full  and I am the first process owning only
     *    ilocblk full blocks of size NB, that is N = INB+(nblocks-1)*NB and
     *    Np = ilocblk * NB = N - INB + (ilocblk-nblocks+1) * NB.
     */
    if(nblocks < NPROCS)
      return ((mydist < nblocks)
                  ? NB
                  : ((mydist > nblocks) ? 0 : N - inb + NB * (1 - nblocks)));

    ilocblk = nblocks / NPROCS;
    mydist -= nblocks - ilocblk * NPROCS;
    return ((mydist < 0)
                ? (ilocblk + 1) * NB
                : ((mydist > 0) ? ilocblk * NB
                                : N - inb + NB * (ilocblk - nblocks + 1)));
  }
}
