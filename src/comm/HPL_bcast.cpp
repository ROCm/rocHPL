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

int HPL_bcast(double*   SBUF,
              int       SCOUNT,
              int       ROOT,
              MPI_Comm  COMM,
              HPL_T_TOP top) {
  /*
   * Purpose
   * =======
   *
   * HPL_bcast is a simple wrapper around  MPI_Bcast.  Its  main  purpose is
   * to  allow for some  experimentation / tuning  of this simple routine.
   * Successful  completion  is  indicated  by  the  returned  error  code
   * HPL_SUCCESS.  In the case of messages of length less than or equal to
   * zero, this function returns immediately.
   *
   * Arguments
   * =========
   *
   * SBUF    (local input)                 double *
   *         On entry, SBUF specifies the starting address of buffer to be
   *         broadcast.
   *
   * SCOUNT  (local input)                 int
   *         On entry,  SCOUNT  specifies  the number of  double precision
   *         entries in SBUF. SCOUNT must be at least zero.
   *
   * ROOT    (local input)                 int
   *         On entry, ROOT specifies the rank of the origin process in
   *         the communication space defined by COMM.
   *
   * COMM    (local input)                 MPI_Comm
   *         The MPI communicator identifying the communication space.
   *
   * ---------------------------------------------------------------------
   */

  if(SCOUNT <= 0) return (HPL_SUCCESS);

  int ierr;

  HPL_TracingPush("HPL_Bcast");

#ifdef HPL_USE_COLLECTIVES

  ierr = MPI_Bcast(SBUF, SCOUNT, MPI_DOUBLE, ROOT, COMM);

#else

  switch(top) {
    case HPL_1RING_M: ierr = HPL_bcast_1rinM(SBUF, SCOUNT, ROOT, COMM); break;
    case HPL_1RING: ierr = HPL_bcast_1ring(SBUF, SCOUNT, ROOT, COMM); break;
    case HPL_2RING_M: ierr = HPL_bcast_2rinM(SBUF, SCOUNT, ROOT, COMM); break;
    case HPL_2RING: ierr = HPL_bcast_2ring(SBUF, SCOUNT, ROOT, COMM); break;
    case HPL_BLONG_M: ierr = HPL_bcast_blonM(SBUF, SCOUNT, ROOT, COMM); break;
    case HPL_BLONG: ierr = HPL_bcast_blong(SBUF, SCOUNT, ROOT, COMM); break;
    default: ierr = HPL_FAILURE;
  }

#endif

  HPL_TracingPop();

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
