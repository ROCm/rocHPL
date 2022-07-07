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

void HPL_pipid(HPL_T_panel* PANEL, int* K, int* IPID) {
  /*
   * Purpose
   * =======
   *
   * HPL_pipid computes an array  IPID  that contains the source and final
   * destination  of  matrix rows  resulting  from  the  application  of N
   * interchanges  as computed by the  LU  factorization  with row partial
   * pivoting. The array IPID is such that the row of global index IPID(i)
   * should be mapped onto the row of global index IPID(i+1). Note that we
   * cannot really know the length of IPID a priori. However, we know that
   * this array is at least 2*N long,  since  there are N rows to swap and
   * broadcast. The length of this array  must be smaller than or equal to
   * 4*N, since every row is swapped with at most a single distinct remote
   * row. The algorithm constructing  IPID  goes as follows: Let IA be the
   * global index of the first row to be swapped.
   *
   * For every row src IA + i with i in [0..N) to be swapped with row  dst
   * such that dst is given by DPIV[i]:
   *
   * Is row  src  the destination  of a previous row of the current block,
   * that is, is there k odd such that IPID(k) is equal to src ?
   *     Yes:  update  this destination  with dst.  For  example,  if  the
   * pivot array is  (0,2)(1,1)(2,5) ... , then when we swap rows 2 and 5,
   * we swap in fact row 0 and 5,  i.e.,  row 0 goes to 5 and not 2  as it
   * was thought so far ...
   *     No :  add  the pair (src,dst) at the end of IPID; row src has not
   * been moved yet.
   *
   * Is row  dst  different  from src the destination of a previous row of
   * the current block, i.e., is there k odd such that IPID(k) is equal to
   * dst ?
   *     Yes:  update  IPID(k) with src.  For example,  if the pivot array
   * is (0,5)(1,1)(2,5) ... , then when  we swap rows  2 and 5, we swap in
   * fact row 2 and 0,  i.e.,  row 0 goes to 2 and not 5 as it was thought
   * so far ...
   *     No : add  the  pair (dst,src) at the end of IPID; row dst has not
   * been moved yet.
   *
   * Note that when src is equal to dst, the pair (dst,src)  should not be
   * added to  IPID  in  order  to avoid duplicated entries in this array.
   * During  the construction of the array  IPID,  we  make  sure that the
   * first N entries are such that IPID(k) with k odd is equal to  IA+k/2.
   * For k in  [0..K/2),  the  row  of global index  IPID(2*k)  should  be
   * mapped onto the row of global index IPID(2*k+1).
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * K       (global output)               int *
   *         On exit, K specifies the number of entries in  IPID.  K is at
   *         least 2*N, and at most 4*N.
   *
   * IPID    (global output)               int *
   *         On entry, IPID is an array of length 4*N.  On exit, the first
   *         K entries of that array contain the src and final destination
   *         resulting  from  the  application of the  N  interchanges  as
   *         specified by  DPIV.  The  pairs  (src,dst)  are  contiguously
   *         stored and sorted so that IPID(2*i+1) is equal to IA+i with i
   *         in [0..N)
   *
   * ---------------------------------------------------------------------
   */

  int  dst, fndd, fnds, ia, i, j, jb, lst, off, src;
  int* ipiv;

  ipiv = PANEL->ipiv;
  jb   = PANEL->jb;
  src = ia = PANEL->ia;
  dst      = (int)(ipiv[0]);
  IPID[0]  = dst;
  IPID[1]  = src;
  *K       = 2;
  if(src != dst) {
    IPID[2] = src;
    IPID[3] = dst;
    *K += 2;
  }

  for(i = 1; i < jb; i++) {
    fnds = 0;
    j    = 1;

    if((src = ia + i) == (dst = (int)(ipiv[i]))) {
      do {
        if(src == IPID[j]) {
          fnds = j;
        } else {
          j += 2;
        }
      } while(!(fnds) && (j < *K));
      if(!fnds) {
        lst       = *K;
        off       = 2;
        IPID[lst] = src;
      } else {
        lst = fnds - 1;
        off = 0;
      }
      IPID[lst + 1] = dst;
    } else {
      fndd = 0;
      do {
        if(src == IPID[j]) {
          fnds = j;
        } else if(dst == IPID[j]) {
          fndd = j;
        }
        j += 2;
      } while((!(fnds) || !(fndd)) && (j < *K));
      if(!fnds) {
        IPID[*K]     = src;
        IPID[*K + 1] = dst;
        off          = 2;
      } else {
        IPID[fnds] = dst;
        off        = 0;
      }
      if(!fndd) {
        lst       = *K + off;
        IPID[lst] = dst;
        off += 2;
      } else {
        lst = fndd - 1;
      }
      IPID[lst + 1] = src;
    }
    /*
     * Enforce IPID(1,i) equal to src = ia + i
     */
    if(lst != (j = (i << 1))) {
      src           = IPID[j];
      IPID[j]       = IPID[lst];
      IPID[lst]     = src;
      dst           = IPID[j + 1];
      IPID[j + 1]   = IPID[lst + 1];
      IPID[lst + 1] = dst;
    }
    *K += off;
  }
}
