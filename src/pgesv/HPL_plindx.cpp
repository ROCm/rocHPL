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

void HPL_plindx(HPL_T_panel* PANEL,
                const int    K,
                const int*   IPID,
                int*         IPA,
                int*         LINDXU,
                int*         LINDXAU,
                int*         LINDXA,
                int*         IPLEN,
                int*         PERMU,
                int*         IWORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_plindx computes three local arrays LINDXU, LINDXA, and  LINDXAU
   * containing the  local  source and final destination position  resulting
   * from the application of row interchanges.  In addition, this function
   * computes the array IPLEN that contains the mapping information for the
   * spreading phase.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * K       (global input)                const int
   *         On entry, K specifies the number of entries in IPID.  K is at
   *         least 2*N, and at most 4*N.
   *
   * IPID    (global input)                const int *
   *         On entry,  IPID  is an array of length K. The first K entries
   *         of that array contain the src and final destination resulting
   *         from the application of the interchanges.
   *
   * IPA     (global output)               int *
   *         On exit,  IPA  specifies  the number of rows that the current
   *         process row has that should be swapped with local rows of A.
   *
   * LINDXU  (global output)               int *
   *         On entry, LINDXU  is an array of dimension N. On exit, this
   *         array contains the local indexes of the rows of A I have that
   *         should be copied into U.
   *
   * LINDXAU (global output)               int *
   *         On entry, LINDXAU is an array of dimension N. On exit, this
   *         array contains the local source indexes of the rows of A I
   *         have that should be swapped locally.
   *
   * LINDXA  (global output)               int *
   *         On entry, LINDXA  is an array of dimension N. On exit, this
   *         array contains  the local destination indexes of the rows
   *         of A I have that should be swapped locally.
   *
   * IPLEN   (global output)               int *
   *         On entry, IPLEN is an array of dimension NPROW + 1. On  exit,
   *         this array is such that  IPLEN[i]  is the number of rows of A
   *         in  the  processes  before  process  IPMAP[i]  after the sort
   *         with the convention that IPLEN[nprow]  is the total number of
   *         rows of the panel.  In other words IPLEN[i+1]-IPLEN[i] is the
   *         local number of rows of A that should be moved to the process
   *         IPMAP[i]. IPLEN is such that the number of rows of the source
   *         process  row can be computed as  IPLEN[1] - IPLEN[0], and the
   *         remaining  entries  of  this  array  are  sorted  so that the
   *         quantities IPLEN[i+1] - IPLEN[i] are logarithmically sorted.
   *
   * PERMU   (global output)               int *
   *         On entry,  PERMU  is an array of dimension JB. On exit, PERMU
   *         contains  a sequence of permutations,  that should be applied
   *         in increasing order to permute in place the row panel U.
   *
   * IWORK   (workspace)                   int *
   *         On entry, IWORK is a workarray of dimension 2*JB.
   *
   * ---------------------------------------------------------------------
   */
  const int myrow   = PANEL->grid->myrow;
  const int nprow   = PANEL->grid->nprow;
  const int jb      = PANEL->jb;
  const int nb      = PANEL->nb;
  const int ia      = PANEL->ia;
  const int iroff   = PANEL->ii;
  const int icurrow = PANEL->prow;

  int* iwork = IWORK + jb;

  /*
   * Compute IPLEN
   */
  HPL_piplen(PANEL, K, IPID, IPLEN, IWORK);

  /*
   * Compute the local arrays  LINDXA  and  LINDXAU  containing  the local
   * source and final destination position resulting from  the application
   * of N interchanges. Compute LINDXA and LINDXAU in icurrow,  and LINDXA
   * elsewhere and PERMU in every process.
   */
  if(myrow == icurrow) {
    // for all rows to be swapped
    int ip = 0, ipU = 0;
    for(int i = 0; i < K; i += 2) {
      const int src = IPID[i];
      int       srcrow;
      Mindxg2p(src, nb, nb, srcrow, 0, nprow);

      if(srcrow == icurrow) { // if I own the src row
        const int dst = IPID[i + 1];
        int       dstrow;
        Mindxg2p(dst, nb, nb, dstrow, 0, nprow);

        int il;
        Mindxg2l(il, src, nb, nb, myrow, 0, nprow);

        if((dstrow == icurrow) && (dst - ia < jb)) {
          // if I own the dst and it's in U

          PERMU[ipU] = dst - ia;      // row index in U
          iwork[ipU] = IPLEN[dstrow]; // Index in AllGathered U
          ipU++;

          LINDXU[IPLEN[dstrow]] = il - iroff; // Index in AllGathered U
          IPLEN[dstrow]++;
        } else if(dstrow != icurrow) {
          // else if I don't own the dst

          // Find the IPID pair with dst as the source
          int j = 0;
          int fndd;
          do {
            fndd = (dst == IPID[j]);
            j += 2;
          } while(!fndd && (j < K));
          // This pair must have dst being sent to a position in U

          PERMU[ipU] = IPID[j - 1] - ia; // row index in U
          iwork[ipU] = IPLEN[dstrow];    // Index in AllGathered U
          ipU++;

          LINDXU[IPLEN[dstrow]] = il - iroff; // Index in AllGathered U
          IPLEN[dstrow]++;
        } else if((dstrow == icurrow) && (dst - ia >= jb)) {
          // else I own the dst, but it's not in U

          LINDXAU[ip] = il - iroff; // the src row must be in the first jb rows

          int il;
          Mindxg2l(il, dst, nb, nb, myrow, 0, nprow);
          LINDXA[ip] = il - iroff; // the dst is somewhere below
          ip++;
        }
      }
    }
    *IPA = ip;
  } else {
    // for all rows to be swapped
    int ip = 0, ipU = 0;
    for(int i = 0; i < K; i += 2) {
      const int src = IPID[i];
      int       srcrow;
      Mindxg2p(src, nb, nb, srcrow, 0, nprow);
      const int dst = IPID[i + 1];
      int       dstrow;
      Mindxg2p(dst, nb, nb, dstrow, 0, nprow);
      /*
       * LINDXU[i] is the local index of the row of A that belongs into U
       */
      if(myrow == dstrow) { // if I own the dst row
        int il;
        Mindxg2l(il, dst, nb, nb, myrow, 0, nprow);
        LINDXU[ip] = il - iroff; // Local A index of incoming row
        ip++;
      }
      /*
       * iwork[i] is the local (current) position  index in U
       * PERMU[i] is the local (final) destination index in U
       */

      // if the src row is coming from the current row rank
      if(srcrow == icurrow) {

        if((dstrow == icurrow) && (dst - ia < jb)) {
          // If the row is going into U
          PERMU[ipU] = dst - ia;      // row index in U
          iwork[ipU] = IPLEN[dstrow]; // Index in AllGathered U
          IPLEN[dstrow]++;
          ipU++;
        } else if(dstrow != icurrow) {
          // If the row is going to another rank
          // (So src must be in U)

          // Find the IPID pair with dst as the source
          int j = 0;
          int fndd;
          do {
            fndd = (dst == IPID[j]);
            j += 2;
          } while(!fndd && (j < K));
          // This pair must have dst being sent to a position in U

          PERMU[ipU] = IPID[j - 1] - ia; // row index in U
          iwork[ipU] = IPLEN[dstrow];    // Index in AllGathered U
          IPLEN[dstrow]++;
          ipU++;
        }
      }
    }
    *IPA = 0;
  }
  /*
   * Simplify iwork and PERMU, return in PERMU the sequence of permutation
   * that need to be apply to U after it has been broadcast.
   */
  HPL_perm(jb, iwork, PERMU, IWORK);
  /*
   * Reset IPLEN to its correct value
   */
  for(int i = nprow; i > 0; i--) IPLEN[i] = IPLEN[i - 1];
  IPLEN[0] = 0;
}
