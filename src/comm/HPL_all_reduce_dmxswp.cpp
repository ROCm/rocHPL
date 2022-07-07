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
#include <assert.h>

/* MPI_Op_create is called in main to bind HPL_dmxswp to this MPI_Op */
MPI_Op       HPL_DMXSWP;
MPI_Datatype PDFACT_ROW;

/* Swap-broadcast comparison function usable in MPI_Allreduce */
void HPL_dmxswp(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {

  assert(*datatype == PDFACT_ROW);
  assert(*len == 1);

  int N;
  MPI_Type_size(PDFACT_ROW, &N);

  double* Wwork = static_cast<double*>(invec);
  double* WORK  = static_cast<double*>(inoutvec);

  const int jb = (N / sizeof(double)) - 4;

  // check max column value and overwirte row if new max is found
  const double gmax = Mabs(WORK[0]);
  const double tmp1 = Mabs(Wwork[0]);
  if((tmp1 > gmax) || ((tmp1 == gmax) && (Wwork[3] < WORK[3]))) {
    HPL_dcopy(jb + 4, Wwork, 1, WORK, 1);
  }
}

void HPL_all_reduce_dmxswp(double*   BUFFER,
                           const int COUNT,
                           const int ROOT,
                           MPI_Comm  COMM,
                           double*   WORK) {
  /*
   * Purpose
   * =======
   *
   * HPL_all_reduce_dmxswp is a specialized all_reduce that performs
   * the swap-broadcast of rows.
   *
   * Arguments
   * =========
   *
   * BUFFER  (local input/global output)   double *
   *         On entry,  BUFFER  points to  the  buffer to be combined.  On
   *         exit, this array contains the combined data and  is identical
   *         on all processes in the group.
   *
   * COUNT   (global input)                const int
   *         On entry,  COUNT  indicates the number of entries in  BUFFER.
   *         COUNT must be 4+2*JB, where JB is the length of the rows being
   *         swapped.
   *
   * ROOT    (local input)                 int
   *         On entry, ROOT specifies the rank of the process owning the
   *         row to be swapped.
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * WORK    (local workspace)             double *
   *         On entry, WORK  is a workarray of size at least COUNT.

   * ---------------------------------------------------------------------
   */

  roctxRangePush("HPL_all_reduce_dmxswp");

#ifdef HPL_USE_COLLECTIVES

  const int myrow = static_cast<int>(BUFFER[3]);
  const int jb    = (COUNT - 4) / 2;

  /* Use a normal all_reduce */
  (void)MPI_Allreduce(MPI_IN_PLACE, BUFFER, 1, PDFACT_ROW, HPL_DMXSWP, COMM);

  /*Location of max row*/
  const int maxrow = static_cast<int>(BUFFER[3]);

  if(myrow == ROOT) { /*Root send top row to maxrow*/
    if(maxrow != ROOT) {
      double* Wwork = BUFFER + 4 + jb;
      HPL_send(Wwork, jb, maxrow, MSGID_BEGIN_PFACT, COMM);
    }
  } else if(myrow == maxrow) { /*Recv top row from ROOT*/
    double* Wwork = BUFFER + 4 + jb;
    HPL_recv(Wwork, jb, ROOT, MSGID_BEGIN_PFACT, COMM);
  }

#else

  double       gmax, tmp1;
  double *     A0, *Wmx;
  unsigned int hdim, ip2, ip2_, ipow, k, mask;
  int Np2, cnt_, cnt0, i, icurrow, mydist, mydis_, myrow, n0, nprow, partner,
      rcnt, root, scnt, size_;

  MPI_Comm_rank(COMM, &myrow);
  MPI_Comm_size(COMM, &nprow);

  /*
   * ip2   : largest power of two <= nprow;
   * hdim  : ip2 procs hypercube dim;
   */
  hdim = 0;
  ip2  = 1;
  k    = nprow;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    hdim++;
  }

  n0      = (COUNT - 4) / 2;
  icurrow = ROOT;
  Np2     = (int)((size_ = nprow - ip2) != 0);
  mydist  = MModSub(myrow, icurrow, nprow);

  /*
   * Set up pointers in workspace:  WORK and Wwork  point to the beginning
   * of the buffers of size 4 + 2*N0 to be combined. Wmx points to the row
   * owning the local (before combine) and global (after combine) absolute
   * value max. A0 points to the copy of the current row of the matrix.
   */

  cnt0 = (cnt_ = n0 + 4) + n0;
  A0   = (Wmx = BUFFER + 4) + n0;

  /*
   * Combine the results (bi-directional exchange):  the process coordina-
   * tes are relative to icurrow,  this allows to reduce the communication
   * volume when nprow is not a power of 2.
   *
   * When nprow is not a power of 2:  proc[i-ip2] receives local data from
   * proc[i]  for all i in [ip2..nprow).  In addition,  proc[0]  (icurrow)
   * sends to proc[ip2] the current row of A  for later broadcast in procs
   * [ip2..nprow).
   */
  if((Np2 != 0) && ((partner = (int)((unsigned int)(mydist) ^ ip2)) < nprow)) {
    if((mydist & ip2) != 0) {
      if(mydist == (int)(ip2))
        (void)HPL_sdrv(BUFFER,
                       cnt_,
                       MSGID_BEGIN_PFACT,
                       A0,
                       n0,
                       MSGID_BEGIN_PFACT,
                       MModAdd(partner, icurrow, nprow),
                       COMM);
      else
        (void)HPL_send(BUFFER,
                       cnt_,
                       MModAdd(partner, icurrow, nprow),
                       MSGID_BEGIN_PFACT,
                       COMM);
    } else {
      if(mydist == 0)
        (void)HPL_sdrv(A0,
                       n0,
                       MSGID_BEGIN_PFACT,
                       WORK,
                       cnt_,
                       MSGID_BEGIN_PFACT,
                       MModAdd(partner, icurrow, nprow),
                       COMM);
      else
        (void)HPL_recv(WORK,
                       cnt_,
                       MModAdd(partner, icurrow, nprow),
                       MSGID_BEGIN_PFACT,
                       COMM);

      tmp1 = Mabs(WORK[0]);
      gmax = Mabs(BUFFER[0]);
      if((tmp1 > gmax) || ((tmp1 == gmax) && (WORK[3] < BUFFER[3]))) {
        HPL_dcopy(cnt_, WORK, 1, BUFFER, 1);
      }
    }
  }

  if(mydist < (int)(ip2)) {
    /*
     * power of 2 part of the processes collection: processes  [0..ip2)  are
     * combining (binary exchange); proc[0] has two rows to send, but one to
     * receive.  At every step  k  in [0..hdim) of the algorithm,  a process
     * pair exchanging 2 rows is such that  myrow >> k+1 is 0.  Among  those
     * processes the ones  that are sending one more row than  what they are
     * receiving are such that myrow >> k is equal to 0.
     */
    k    = 0;
    ipow = 1;

    while(k < hdim) {
      if(((unsigned int)(mydist) >> (k + 1)) == 0) {
        if(((unsigned int)(mydist) >> k) == 0) {
          scnt = cnt0;
          rcnt = cnt_;
        } else {
          scnt = cnt_;
          rcnt = cnt0;
        }
      } else {
        scnt = rcnt = cnt_;
      }

      partner = (int)((unsigned int)(mydist) ^ ipow);
      (void)HPL_sdrv(BUFFER,
                     scnt,
                     MSGID_BEGIN_PFACT,
                     WORK,
                     rcnt,
                     MSGID_BEGIN_PFACT,
                     MModAdd(partner, icurrow, nprow),
                     COMM);

      tmp1 = Mabs(WORK[0]);
      gmax = Mabs(BUFFER[0]);
      if((tmp1 > gmax) || ((tmp1 == gmax) && (WORK[3] < BUFFER[3]))) {
        HPL_dcopy((rcnt == cnt0 ? cnt0 : cnt_), WORK, 1, BUFFER, 1);
      } else if(rcnt == cnt0) {
        HPL_dcopy(n0, WORK + cnt_, 1, A0, 1);
      }

      ipow <<= 1;
      k++;
    }
  } else if(size_ > 1) {
    /*
     * proc[ip2] broadcast current row of A to procs [ip2+1..nprow).
     */
    k    = (unsigned int)(size_)-1;
    ip2_ = mask = 1;
    while(k > 1) {
      k >>= 1;
      ip2_ <<= 1;
      mask <<= 1;
      mask++;
    }

    root   = MModAdd(icurrow, (int)(ip2), nprow);
    mydis_ = MModSub(myrow, root, nprow);

    do {
      mask ^= ip2_;
      if((mydis_ & mask) == 0) {
        partner = (int)(mydis_ ^ ip2_);
        if((mydis_ & ip2_) != 0) {
          (void)HPL_recv(
              A0, n0, MModAdd(root, partner, nprow), MSGID_BEGIN_PFACT, COMM);
        } else if(partner < size_) {
          (void)HPL_send(
              A0, n0, MModAdd(root, partner, nprow), MSGID_BEGIN_PFACT, COMM);
        }
      }
      ip2_ >>= 1;
    } while(ip2_ > 0);
  }
  /*
   * If nprow is not a power of 2,  for all i in [ip2..nprow), proc[i-ip2]
   * sends the pivot row to proc[i]  along  with the first four entries of
   * the BUFFER array.
   */
  if((Np2 != 0) && ((partner = (int)((unsigned int)(mydist) ^ ip2)) < nprow)) {
    if((mydist & ip2) != 0) {
      (void)HPL_recv(BUFFER,
                     cnt_,
                     MModAdd(partner, icurrow, nprow),
                     MSGID_BEGIN_PFACT,
                     COMM);
    } else {
      (void)HPL_send(BUFFER,
                     cnt_,
                     MModAdd(partner, icurrow, nprow),
                     MSGID_BEGIN_PFACT,
                     COMM);
    }
  }

#endif
  roctxRangePop();
}
