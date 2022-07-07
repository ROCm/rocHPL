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

int HPL_reduce(void*            BUFFER,
               const int        COUNT,
               const HPL_T_TYPE DTYPE,
               const HPL_T_OP   OP,
               const int        ROOT,
               MPI_Comm         COMM) {
  /*
   * Purpose
   * =======
   *
   * HPL_reduce performs a global reduce operation across all processes of
   * a group.  Note that the input buffer is  used as workarray and in all
   * processes but the accumulating process corrupting the original data.
   *
   * Arguments
   * =========
   *
   * BUFFER  (local input/output)          void *
   *         On entry,  BUFFER  points to  the  buffer to be  reduced.  On
   *         exit,  and  in process of rank  ROOT  this array contains the
   *         reduced data.  This  buffer  is also used as workspace during
   *         the operation in the other processes of the group.
   *
   * COUNT   (global input)                const int
   *         On entry,  COUNT  indicates the number of entries in  BUFFER.
   *         COUNT must be at least zero.
   *
   * DTYPE   (global input)                const HPL_T_TYPE
   *         On entry,  DTYPE  specifies the type of the buffers operands.
   *
   * OP      (global input)                const HPL_T_OP
   *         On entry, OP is a pointer to the local combine function.
   *
   * ROOT    (global input)                const int
   *         On entry, ROOT is the coordinate of the accumulating process.
   *
   * COMM    (global/local input)          MPI_Comm
   *         The MPI communicator identifying the process collection.
   *
   * ---------------------------------------------------------------------
   */

  int ierr;

  int rank;
  MPI_Comm_rank(COMM, &rank);

  if(rank == ROOT)
    ierr = MPI_Reduce(
        MPI_IN_PLACE, BUFFER, COUNT, HPL_2_MPI_TYPE(DTYPE), OP, ROOT, COMM);
  else
    ierr =
        MPI_Reduce(BUFFER, NULL, COUNT, HPL_2_MPI_TYPE(DTYPE), OP, ROOT, COMM);

  return ((ierr == MPI_SUCCESS ? HPL_SUCCESS : HPL_FAILURE));
}
