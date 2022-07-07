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

int HPL_recv(double* RBUF, int RCOUNT, int SRC, int RTAG, MPI_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_recv is a simple wrapper around  MPI_Recv.  Its  main  purpose is
   * to  allow for some  experimentation / tuning  of this simple routine.
   * Successful  completion  is  indicated  by  the  returned  error  code
   * HPL_SUCCESS.  In the case of messages of length less than or equal to
   * zero, this function returns immediately.
   *
   * Arguments
   * =========
   *
   * RBUF    (local output)                double *
   *         On entry, RBUF specifies the starting address of buffer to be
   *         received.
   *
   * RCOUNT  (local input)                 int
   *         On entry,  RCOUNT  specifies  the number  of double precision
   *         entries in RBUF. RCOUNT must be at least zero.
   *
   * SRC     (local input)                 int
   *         On entry, SRC  specifies the rank of the  sending  process in
   *         the communication space defined by COMM.
   *
   * RTAG    (local input)                 int
   *         On entry,  STAG specifies the message tag to be used for this
   *         communication operation.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  if(RCOUNT <= 0) return (HPL_SUCCESS);

  MPI_Status status;

  int ierr =
      MPI_Recv((void*)(RBUF), RCOUNT, MPI_DOUBLE, SRC, RTAG, COMM, &status);

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
