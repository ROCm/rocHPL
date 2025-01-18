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

int HPL_allgatherv(double*    BUF,
                   const int  SCOUNT,
                   const int* RCOUNT,
                   const int* DISPL,
                   MPI_Comm   COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_allgatherv is a simple wrapper around an in-place MPI_Allgatherv.
   * Its  main  purpose is to  allow for some  experimentation / tuning
   * of this simple routine. Successful  completion  is  indicated  by
   * the  returned  error  code HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * BUF    (local input/output)           double *
   *         On entry, on the root process BUF specifies the starting
   *         address of buffer to be gathered.
   *
   * SCOUNT  (local input)                 int
   *         On entry,  SCOUNT is an array of length SIZE specifiying
   *         the number of  double precision entries in BUF to send to
   *         each process.
   *
   * RCOUNT  (local input)                 int
   *         On entry,  RCOUNT is an array of length SIZE specifiying
   *         the number of double precision entries in BUF to receive from
   *         each process.
   *
   * DISPL   (local input)                 int *
   *         On entry,  DISPL is an array of length SIZE specifiying the
   *         displacement (relative to BUF) from which to place the incoming
   *         data from each process.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  HPL_TracingPush("HPL_Allgatherv");

#ifdef HPL_USE_COLLECTIVES

  int ierr = MPI_Allgatherv(
      MPI_IN_PLACE, SCOUNT, MPI_DOUBLE, BUF, RCOUNT, DISPL, MPI_DOUBLE, COMM);

#else

  int rank, size, ierr = MPI_SUCCESS;
  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);

  /*
   * Ring exchange
   */
  const int npm1 = size - 1;
  const int prev = MModSub1(rank, size);
  const int next = MModAdd1(rank, size);

  const int tag = 0;

  for(int k = 0; k < npm1; k++) {
    MPI_Request request;
    MPI_Status  status;
    const int   l = (int)((unsigned int)(k) >> 1);

    int il, lengthS, lengthR, partner, ibufS, ibufR;
    if(((rank + k) & 1) != 0) {
      il      = MModAdd(rank, l, size);
      ibufS   = DISPL[il];
      lengthS = RCOUNT[il];
      il      = MModSub(rank, l + 1, size);
      ibufR   = DISPL[il];
      lengthR = RCOUNT[il];
      partner = prev;
    } else {
      il      = MModSub(rank, l, size);
      ibufS   = DISPL[il];
      lengthS = RCOUNT[il];
      il      = MModAdd(rank, l + 1, size);
      ibufR   = DISPL[il];
      lengthR = RCOUNT[il];
      partner = next;
    }

    if(lengthR > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Irecv(
            BUF + ibufR, lengthR, MPI_DOUBLE, partner, tag, COMM, &request);
    }

    if(lengthS > 0) {
      if(ierr == MPI_SUCCESS)
        ierr = MPI_Send(BUF + ibufS, lengthS, MPI_DOUBLE, partner, tag, COMM);
    }

    if(lengthR > 0) {
      if(ierr == MPI_SUCCESS) ierr = MPI_Wait(&request, &status);
    }
  }

#endif

  HPL_TracingPop();

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
