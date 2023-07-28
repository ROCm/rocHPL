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

#define BLOCK_SIZE 512

__global__ void hpl_randmat(const int      mp,
                            const int      nq,
                            const int      NB,
                            const int      LDA,
                            const uint64_t cblkjumpA,
                            const uint64_t cblkjumpC,
                            const uint64_t rblkjumpA,
                            const uint64_t rblkjumpC,
                            const uint64_t cjumpA,
                            const uint64_t cjumpC,
                            const uint64_t rjumpA,
                            const uint64_t rjumpC,
                            const uint64_t startrand,
                            double* __restrict__ A) {

  const int jblk = blockIdx.y;
  const int iblk = blockIdx.x;

  /* Get panel size */
  const int jb = (jblk == gridDim.y - 1) ? nq - ((nq - 1) / NB) * NB : NB;
  const int ib = (iblk == gridDim.x - 1) ? mp - ((mp - 1) / NB) * NB : NB;

  double* Ab = A + iblk * NB + static_cast<size_t>(jblk * NB) * LDA;

  /* Start at first uint64_t */
  uint64_t irand = startrand;

  /* Jump rand M*NB*npcol for each jblk */
  for(int j = 0; j < jblk; ++j) { irand = cblkjumpA * irand + cblkjumpC; }

  /* Jump rand NB*nprow for each iblk */
  for(int i = 0; i < iblk; ++i) { irand = rblkjumpA * irand + rblkjumpC; }

  /* Shift per-column irand */
  const int n = threadIdx.x;
  for(int j = 0; j < threadIdx.x; ++j) { irand = cjumpA * irand + cjumpC; }

  for(int n = threadIdx.x; n < jb; n += blockDim.x) {
    /*Grab rand at top of block*/
    uint64_t r = irand;

    /* Each thread traverses a column */
    for(int m = 0; m < ib; ++m) {
      /*Generate a random double from the current r */
      const double p1 = ((r & (65535LU << 0)) >> 0);
      const double p2 = ((r & (65535LU << 16)) >> 16);
      const double p3 = ((r & (65535LU << 32)) >> 32);
      const double p4 = ((r & (65535LU << 48)) >> 48);

      Ab[m + n * LDA] =
          (HPL_HALF - (((p1) + (p2)*HPL_POW16) / HPL_DIVFAC * HPL_HALF + (p3) +
                       (p4)*HPL_POW16) /
                          HPL_DIVFAC * HPL_HALF);

      /*Increment rand*/
      r = rjumpA * r + rjumpC;
    }

    /* Block-shift per-column irand */
    for(int j = 0; j < blockDim.x; ++j) { irand = cjumpA * irand + cjumpC; }
  }
}

void HPL_pdrandmat(const HPL_T_grid* GRID,
                   const int         M,
                   const int         N,
                   const int         NB,
                   double*           A,
                   const int         LDA,
                   const int         ISEED) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdrandmat generates (or regenerates) a parallel random matrix A.
   *
   * The  pseudo-random  generator uses the linear congruential algorithm:
   * X(n+1) = (a * X(n) + c) mod m  as  described  in the  Art of Computer
   * Programming, Knuth 1973, Vol. 2.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 const HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * M       (global input)                const int
   *         On entry,  M  specifies  the number  of rows of the matrix A.
   *         M must be at least zero.
   *
   * N       (global input)                const int
   *         On entry,  N specifies the number of columns of the matrix A.
   *         N must be at least zero.
   *
   * NB      (global input)                const int
   *         On entry,  NB specifies the blocking factor used to partition
   *         and distribute the matrix A. NB must be larger than one.
   *
   * A       (local output)                double *
   *         On entry,  A  points  to an array of dimension (LDA,LocQ(N)).
   *         On exit, this array contains the coefficients of the randomly
   *         generated matrix.
   *
   * LDA     (local input)                 const int
   *         On entry, LDA specifies the leading dimension of the array A.
   *         LDA must be at least max(1,LocP(M)).
   *
   * ISEED   (global input)                const int
   *         On entry, ISEED  specifies  the  seed  number to generate the
   *         matrix A. ISEED must be at least zero.
   *
   * ---------------------------------------------------------------------
   */
  int mp, mycol, myrow, npcol, nprow, nq;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  uint64_t mult64  = HPL_MULT;
  uint64_t iadd64  = HPL_IADD;
  uint64_t jseed64 = static_cast<uint64_t>(ISEED);

  /*
   * Generate an M by N matrix starting in process (0,0)
   */
  Mnumroc(mp, M, NB, NB, myrow, 0, nprow);
  Mnumroc(nq, N, NB, NB, mycol, 0, npcol);

  if((mp <= 0) || (nq <= 0)) return;

  /*
   * Compute multiplier/adder for various jumps in random sequence
   */
  const int jump1 = 1;
  const int jump2 = nprow * NB;
  const int jump3 = M;
  const int jump4 = npcol * NB;
  const int jump5 = NB;
  const int jump6 = mycol;
  const int jump7 = myrow * NB;

  uint64_t startrand;
  uint64_t rjumpA, rblkjumpA, cjumpA, cblkjumpA, ia564;
  uint64_t rjumpC, rblkjumpC, cjumpC, cblkjumpC, ic564;
  uint64_t itmp164, itmp264, itmp364;

  /* Compute different jump coefficients */
  HPL_xjumpm(jump1, mult64, iadd64, jseed64, startrand, rjumpA, rjumpC);
  HPL_xjumpm(jump2, mult64, iadd64, startrand, itmp164, rblkjumpA, rblkjumpC);
  HPL_xjumpm(jump3, mult64, iadd64, startrand, itmp164, cjumpA, cjumpC);
  HPL_xjumpm(jump4, cjumpA, cjumpC, startrand, itmp164, cblkjumpA, cblkjumpC);

  /* Shift the starting random value for this rank */
  HPL_xjumpm(jump5, cjumpA, cjumpC, startrand, itmp164, ia564, ic564);
  HPL_xjumpm(jump6, ia564, ic564, startrand, itmp364, itmp164, itmp264);
  HPL_xjumpm(jump7, mult64, iadd64, itmp364, startrand, itmp164, itmp264);

  /*
   * Local number of blocks
   */
  const int mblks = (mp + NB - 1) / NB;
  const int nblks = (nq + NB - 1) / NB;

  /* Initialize on GPU */
  dim3 grid = dim3(mblks, nblks);
  hpl_randmat<<<grid, BLOCK_SIZE>>>(mp,
                                    nq,
                                    NB,
                                    LDA,
                                    cblkjumpA,
                                    cblkjumpC,
                                    rblkjumpA,
                                    rblkjumpC,
                                    cjumpA,
                                    cjumpC,
                                    rjumpA,
                                    rjumpC,
                                    startrand,
                                    A);
  CHECK_HIP_ERROR(hipGetLastError());
  CHECK_HIP_ERROR(hipDeviceSynchronize());
}
