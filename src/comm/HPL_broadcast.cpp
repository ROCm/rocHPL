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

int HPL_broadcast(void*            BUFFER,
                  const int        COUNT,
                  const HPL_T_TYPE DTYPE,
                  const int        ROOT,
                  MPI_Comm         COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_broadcast broadcasts  a message from the process with rank ROOT to
   * all processes in the group.
   *
   * Arguments
   * =========
   *
   * BUFFER  (local input/output)          void *
   *         On entry,  BUFFER  points to  the  buffer to be broadcast. On
   *         exit, this array contains the broadcast data and is identical
   *         on all processes in the group.
   *
   * COUNT   (global input)                const int
   *         On entry,  COUNT  indicates the number of entries in  BUFFER.
   *         COUNT must be at least zero.
   *
   * DTYPE   (global input)                const HPL_T_TYPE
   *         On entry,  DTYPE  specifies the type of the buffers operands.
   *
   * ROOT    (global input)                const int
   *         On entry, ROOT is the coordinate of the source process.
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * ---------------------------------------------------------------------
   */

  int ierr = MPI_Bcast(BUFFER, COUNT, HPL_2_MPI_TYPE(DTYPE), ROOT, COMM);

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
