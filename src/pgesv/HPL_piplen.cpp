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

void HPL_piplen(HPL_T_panel* PANEL,
                const int    K,
                const int*   IPID,
                int*         IPLEN,
                int*         IWORK) {

  const int nprow   = PANEL->grid->nprow;
  const int jb      = PANEL->jb;
  const int nb      = PANEL->nb;
  const int ia      = PANEL->ia;
  const int icurrow = PANEL->prow;

  int* iwork = IWORK + jb;

  /*
   * Compute IPLEN
   */
  for(int i = 0; i <= nprow; i++) IPLEN[i] = 0;

  /*
   * IPLEN[i]  is the number of rows of A in the processes  before
   * process i, with the convention that IPLEN[nprow] is the total
   * number of rows.
   * In other words,  IPLEN[i+1] - IPLEN[i] is the local number of
   * rows of  A  that should be moved for each process.
   */
  for(int i = 0; i < K; i += 2) {
    const int src = IPID[i];
    int       srcrow;
    Mindxg2p(src, nb, nb, srcrow, 0, nprow);
    if(srcrow == icurrow) {
      const int dst = IPID[i + 1];
      int       dstrow;
      Mindxg2p(dst, nb, nb, dstrow, 0, nprow);
      if((dstrow != srcrow) || (dst - ia < jb)) IPLEN[dstrow + 1]++;
    }
  }

  for(int i = 1; i <= nprow; i++) { IPLEN[i] += IPLEN[i - 1]; }
}
