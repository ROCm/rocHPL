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

int HPL_send(double* SBUF, int SCOUNT, int DEST, int STAG, MPI_Comm COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_send is a simple wrapper around  MPI_Send.  Its  main  purpose is
   * to  allow for some  experimentation / tuning  of this simple routine.
   * Successful  completion  is  indicated  by  the  returned  error  code
   * MPI_SUCCESS.  In the case of messages of length less than or equal to
   * zero, this function returns immediately.
   *
   * Arguments
   * =========
   *
   * SBUF    (local input)                 double *
   *         On entry, SBUF specifies the starting address of buffer to be
   *         sent.
   *
   * SCOUNT  (local input)                 int
   *         On entry,  SCOUNT  specifies  the number of  double precision
   *         entries in SBUF. SCOUNT must be at least zero.
   *
   * DEST    (local input)                 int
   *         On entry, DEST specifies the rank of the receiving process in
   *         the communication space defined by COMM.
   *
   * STAG    (local input)                 int
   *         On entry,  STAG specifies the message tag to be used for this
   *         communication operation.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  if(SCOUNT <= 0) return (HPL_SUCCESS);

  int ierr = MPI_Send((void*)(SBUF), SCOUNT, MPI_DOUBLE, DEST, STAG, COMM);

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
