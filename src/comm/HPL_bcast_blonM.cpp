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

int HPL_bcast_blonM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*
   * Cast phase:  ROOT process  sends to its right neighbor,  then spread
   * the panel on the other npcol - 2 processes.  If  I  am  not the ROOT
   * process, probe for message received.  If the message is there,  then
   * receive it. If I am just after the ROOT process, return.  Otherwise,
   * keep spreading on those npcol - 2 processes.  Otherwise,  inform the
   * caller that the panel has still not been received.
   */
  int count, ierr = MPI_SUCCESS, ibuf, ibufR, ibufS, indx, ip2 = 1, k, l, lbuf,
             lbufR, lbufS, mask = 1, mydist, mydist2, next, npm1, npm2, partner,
             prev;

  const int tag = ROOT;
  next          = MModAdd1(rank, size);
  prev          = MModSub1(rank, size);

  if(rank == ROOT) {
    if(ierr == MPI_SUCCESS)
      ierr =
          MPI_Send(SBUF, SCOUNT, MPI_DOUBLE, MModAdd1(rank, size), tag, COMM);
  } else if(prev == ROOT) {
    if(ierr == MPI_SUCCESS)
      ierr = MPI_Recv(
          SBUF, SCOUNT, MPI_DOUBLE, ROOT, tag, COMM, MPI_STATUS_IGNORE);
  }
  /*
   * if I am just after the ROOT, exit now. The message receive  completed
   * successfully, this guy is done. If there are only 2 processes in each
   * row of processes, we are done as well.
   */
  if((prev == ROOT) || (size == 2)) return ierr;
  /*
   * Otherwise, proceed with broadcast -  Spread  the panel across process
   * columns
   */
  npm2 = (npm1 = size - 1) - 1;

  k = npm2;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    mask <<= 1;
    mask++;
  }
  if(rank == ROOT)
    mydist2 = (mydist = 0);
  else
    mydist2 = (mydist = MModSub(rank, ROOT, size) - 1);

  indx  = ip2;
  count = SCOUNT / npm1;
  count = Mmax(count, 1);

  do {
    mask ^= ip2;

    if((mydist & mask) == 0) {
      lbuf = SCOUNT - (ibuf = indx * count);
      if(indx + ip2 < npm1) {
        l    = ip2 * count;
        lbuf = Mmin(lbuf, l);
      }

      partner = mydist ^ ip2;

      if((mydist & ip2) != 0) {
        partner = MModAdd(ROOT, partner, size);
        if(partner != ROOT) partner = MModAdd1(partner, size);

        if(lbuf > 0) {
          if(ierr == MPI_SUCCESS)
            ierr = MPI_Recv(SBUF + ibuf,
                            lbuf,
                            MPI_DOUBLE,
                            partner,
                            tag,
                            COMM,
                            MPI_STATUS_IGNORE);
        }
      } else if(partner < npm1) {
        partner = MModAdd(ROOT, partner, size);
        if(partner != ROOT) partner = MModAdd1(partner, size);

        if(lbuf > 0) {
          if(ierr == MPI_SUCCESS)
            ierr = MPI_Send(SBUF + ibuf, lbuf, MPI_DOUBLE, partner, tag, COMM);
        }
      }
    }

    if(mydist2 < ip2) {
      ip2 >>= 1;
      indx -= ip2;
    } else {
      mydist2 -= ip2;
      ip2 >>= 1;
      indx += ip2;
    }

  } while(ip2 > 0);
  /*
   * Roll the pieces
   */
  if(MModSub1(prev, size) == ROOT) prev = ROOT;
  if(rank == ROOT) next = MModAdd1(next, size);

  for(k = 0; k < npm2; k++) {
    l = (k >> 1);
    /*
     * Who is sending to who and how much
     */
    if(((mydist + k) & 1) != 0) {
      ibufS = (indx = MModAdd(mydist, l, npm1)) * count;
      lbufS = (indx == npm2 ? SCOUNT : ibufS + count);
      lbufS = Mmin(SCOUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModSub(mydist, l + 1, npm1)) * count;
      lbufR = (indx == npm2 ? SCOUNT : ibufR + count);
      lbufR = Mmin(SCOUNT, lbufR) - ibufR;
      lbufR = Mmax(0, lbufR);

      partner = prev;
    } else {
      ibufS = (indx = MModSub(mydist, l, npm1)) * count;
      lbufS = (indx == npm2 ? SCOUNT : ibufS + count);
      lbufS = Mmin(SCOUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModAdd(mydist, l + 1, npm1)) * count;
      lbufR = (indx == npm2 ? SCOUNT : ibufR + count);
      lbufR = Mmin(SCOUNT, lbufR) - ibufR;
      lbufR = Mmax(0, lbufR);

      partner = next;
    }
    /*
     * Exchange the messages
     */
    MPI_Request request;
    MPI_Status  status;

    if(lbufR > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Irecv(
            SBUF + ibufR, lbufR, MPI_DOUBLE, partner, tag, COMM, &request);
    }

    if(lbufS > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Send(SBUF + ibufS, lbufS, MPI_DOUBLE, partner, tag, COMM);
    }

    if(lbufR > 0)
      if(ierr == MPI_SUCCESS) ierr = MPI_Wait(&request, &status);
  }

  return ierr;
}
