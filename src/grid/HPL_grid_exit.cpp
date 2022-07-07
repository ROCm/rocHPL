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

int HPL_grid_exit(HPL_T_grid* GRID) {
  /*
   * Purpose
   * =======
   *
   * HPL_grid_exit marks  the process  grid object for  deallocation.  The
   * returned  error  code  MPI_SUCCESS  indicates  successful completion.
   * Other error codes are (MPI) implementation dependent.
   *
   * Arguments
   * =========
   *
   * GRID    (local input/output)          HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid to be released.
   *
   * ---------------------------------------------------------------------
   */

  int hplerr = MPI_SUCCESS, mpierr;

  if(GRID->all_comm != MPI_COMM_NULL) {
    mpierr = MPI_Comm_free(&(GRID->row_comm));
    if(mpierr != MPI_SUCCESS) hplerr = mpierr;
    mpierr = MPI_Comm_free(&(GRID->col_comm));
    if(mpierr != MPI_SUCCESS) hplerr = mpierr;
    mpierr = MPI_Comm_free(&(GRID->all_comm));
    if(mpierr != MPI_SUCCESS) hplerr = mpierr;
  }

  GRID->order = HPL_COLUMN_MAJOR;

  GRID->iam = GRID->myrow = GRID->mycol = -1;
  GRID->nprow = GRID->npcol = GRID->nprocs = -1;

  GRID->row_ip2 = GRID->row_hdim = GRID->row_ip2m1 = GRID->row_mask = -1;
  GRID->col_ip2 = GRID->col_hdim = GRID->col_ip2m1 = GRID->col_mask = -1;

  return (hplerr);
}
