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

int HPL_bcast_2ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM) {

  int rank, size;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  if(size <= 1) return (MPI_SUCCESS);

  /*One ring exchange to rule them all*/
  int chunk_size = 512 * 512; // 2MB

  chunk_size = std::min(chunk_size, SCOUNT);

  MPI_Request request[4];

  request[0] = MPI_REQUEST_NULL;
  request[1] = MPI_REQUEST_NULL;
  request[2] = MPI_REQUEST_NULL;
  request[3] = MPI_REQUEST_NULL;

  const int Nchunks     = (SCOUNT + chunk_size - 1) / chunk_size;
  const int NchunksHalf = (Nchunks + 1) / 2;

  const int tag  = rank;
  const int next = MModAdd1(rank, size);
  const int prev = MModSub1(rank, size);

  /*Mid point of message*/
  double* SBUF0 = SBUF;
  double* SBUF1 = SBUF + NchunksHalf * chunk_size;

  double* RBUF0 = SBUF0;
  double* RBUF1 = SBUF1;

  /*Shift to ROOT=0*/
  rank = MModSub(rank, ROOT, size);

  int Nsend0 = (rank == size - 1) ? 0 : NchunksHalf * chunk_size;
  int Nsend1 = (rank == 1) ? 0 : SCOUNT - NchunksHalf * chunk_size;

  int Nrecv0 = (rank == 0) ? 0 : NchunksHalf * chunk_size;
  int Nrecv1 = (rank == 0) ? 0 : SCOUNT - NchunksHalf * chunk_size;

  /*Recv from left*/
  int Nr0 = std::min(Nrecv0, chunk_size);
  if(Nr0 > 0) {
    MPI_Irecv(RBUF0, Nr0, MPI_DOUBLE, prev, prev, COMM, request + 0);
  }

  /*Recv from right*/
  int Nr1 = std::min(Nrecv1, chunk_size);
  if(Nr1 > 0) {
    MPI_Irecv(RBUF1, Nr1, MPI_DOUBLE, next, next, COMM, request + 1);
  }

  /*Send to right if there is data present to send*/
  int Ns0 = std::min(Nsend0 - Nrecv0, chunk_size);
  if(Ns0 > 0) {
    MPI_Isend(SBUF0, Ns0, MPI_DOUBLE, next, tag, COMM, request + 2);
  }

  /*Send to left if there is data present to send*/
  int Ns1 = std::min(Nsend1 - Nrecv1, chunk_size);
  if(Ns1 > 0) {
    MPI_Isend(SBUF1, Ns1, MPI_DOUBLE, prev, tag, COMM, request + 3);
  }

  while(Nsend0 > 0 || Nsend1 > 0 || Nrecv0 > 0 || Nrecv1 > 0) {
    int index = -1;
    MPI_Waitany(4, request, &index, MPI_STATUSES_IGNORE);

    if(index == 0) { /*Recv'd from left*/
      /*If we're waiting on this recv in order to send, send now*/
      if(Nrecv0 == Nsend0) {
        Ns0 = Nr0;
        MPI_Isend(SBUF0, Ns0, MPI_DOUBLE, next, tag, COMM, request + 2);
      }

      /*Count the recv'd amounts*/
      Nrecv0 -= Nr0;
      RBUF0 += Nr0;

      /*Post next recv if needed*/
      Nr0 = std::min(Nrecv0, chunk_size);
      if(Nr0 > 0) {
        MPI_Irecv(RBUF0, Nr0, MPI_DOUBLE, prev, prev, COMM, request + 0);
      } else {
        request[0] = MPI_REQUEST_NULL;
      }

    } else if(index == 1) { /*Recv'd from right*/
      /*If we're waiting on this recv in order to send, send now*/
      if(Nrecv1 == Nsend1) {
        Ns1 = Nr1;
        MPI_Isend(SBUF1, Ns1, MPI_DOUBLE, prev, tag, COMM, request + 3);
      }

      /*Count the recv'd amounts*/
      Nrecv1 -= Nr1;
      RBUF1 += Nr1;

      /*Post next recv if needed*/
      Nr1 = std::min(Nrecv1, chunk_size);
      if(Nr1 > 0) {
        MPI_Irecv(RBUF1, Nr1, MPI_DOUBLE, next, next, COMM, request + 1);
      } else {
        request[1] = MPI_REQUEST_NULL;
      }

    } else if(index == 2) { /*Sent to right */
      Nsend0 -= Ns0;
      SBUF0 += Ns0;

      /*Send to right if there is data present to send*/
      Ns0 = std::min(Nsend0 - Nrecv0, chunk_size);
      if(Ns0 > 0) {
        MPI_Isend(SBUF0, Ns0, MPI_DOUBLE, next, tag, COMM, request + 2);
      } else {
        request[2] = MPI_REQUEST_NULL;
      }
    } else { /*index==3, Sent to left */
      Nsend1 -= Ns1;
      SBUF1 += Ns1;

      Ns1 = std::min(Nsend1 - Nrecv1, chunk_size);
      if(Ns1 > 0) {
        MPI_Isend(SBUF1, Ns1, MPI_DOUBLE, prev, tag, COMM, request + 3);
      } else {
        request[3] = MPI_REQUEST_NULL;
      }
    }
  }

  return MPI_SUCCESS;
}
