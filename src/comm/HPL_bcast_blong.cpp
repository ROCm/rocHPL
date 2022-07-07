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

int HPL_bcast_blong(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*
   * Cast phase:  If I am the ROOT process, start spreading the panel.  If
   * I am not the ROOT process,  test  for  message receive completion. If
   * the message  is there,  then receive it,  and  keep  spreading  in  a
   * blocking fashion this time.  Otherwise,  inform  the caller  that the
   * panel has still not been received.
   */
  int count, ierr = MPI_SUCCESS, ibuf, ibufR, ibufS, indx, ip2, k, l, lbuf,
             lbufR, lbufS, mask, mydist, mydist2, npm1, partner, next, prev;

  const int tag = 0;

  // ip2  : largest power of two <= size-1;
  // mask : ip2 procs hypercube mask;
  mask = ip2 = 1;
  k          = size - 1;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    mask <<= 1;
    mask++;
  }

  npm1    = size - 1;
  mydist2 = (mydist = MModSub(rank, ROOT, size));
  indx    = ip2;
  count   = SCOUNT / size;
  count   = Mmax(count, 1);
  /*
   * Spread the panel across process columns
   */
  do {
    mask ^= ip2;

    if((mydist & mask) == 0) {
      lbuf = SCOUNT - (ibuf = indx * count);
      if(indx + ip2 < size) {
        l    = ip2 * count;
        lbuf = Mmin(lbuf, l);
      }

      partner = mydist ^ ip2;

      if((mydist & ip2) != 0) {
        partner = MModAdd(ROOT, partner, size);

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
      } else if(partner < size) {
        partner = MModAdd(ROOT, partner, size);

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
  prev = MModSub1(rank, size);
  next = MModAdd1(rank, size);

  for(k = 0; k < npm1; k++) {
    l = (k >> 1);
    /*
     * Who is sending to who and how much
     */
    if(((mydist + k) & 1) != 0) {
      ibufS = (indx = MModAdd(mydist, l, size)) * count;
      lbufS = (indx == npm1 ? SCOUNT : ibufS + count);
      lbufS = Mmin(SCOUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModSub(mydist, l + 1, size)) * count;
      lbufR = (indx == npm1 ? SCOUNT : ibufR + count);
      lbufR = Mmin(SCOUNT, lbufR) - ibufR;
      lbufR = Mmax(0, lbufR);

      partner = prev;
    } else {
      ibufS = (indx = MModSub(mydist, l, size)) * count;
      lbufS = (indx == npm1 ? SCOUNT : ibufS + count);
      lbufS = Mmin(SCOUNT, lbufS) - ibufS;
      lbufS = Mmax(0, lbufS);

      ibufR = (indx = MModAdd(mydist, l + 1, size)) * count;
      lbufR = (indx == npm1 ? SCOUNT : ibufR + count);
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
