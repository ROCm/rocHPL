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

int HPL_grid_init(MPI_Comm          COMM,
                  const HPL_T_ORDER ORDER,
                  const int         NPROW,
                  const int         NPCOL,
                  const int         p,
                  const int         q,
                  HPL_T_grid*       GRID) {
  /*
   * Purpose
   * =======
   *
   * HPL_grid_init creates a NPROW x NPCOL  process  grid using column- or
   * row-major ordering from an initial collection of processes identified
   * by an  MPI  communicator.  Successful  completion is indicated by the
   * returned error code MPI_SUCCESS.  Other error codes depend on the MPI
   * implementation. The coordinates of processes that are not part of the
   * grid are set to values outside of [0..NPROW) x [0..NPCOL).
   *
   * Arguments
   * =========
   *
   * COMM    (global/local input)          MPI_Comm
   *         On entry,  COMM  is  the  MPI  communicator  identifying  the
   *         initial  collection  of  processes out of which  the  grid is
   *         formed.
   *
   * ORDER   (global input)                const HPL_T_ORDER
   *         On entry, ORDER specifies how the processes should be ordered
   *         in the grid as follows:
   *            ORDER = HPL_ROW_MAJOR    row-major    ordering;
   *            ORDER = HPL_COLUMN_MAJOR column-major ordering;
   *
   * NPROW   (global input)                const int
   *         On entry,  NPROW  specifies the number of process rows in the
   *         grid to be created. NPROW must be at least one.
   *
   * NPCOL   (global input)                const int
   *         On entry,  NPCOL  specifies  the number of process columns in
   *         the grid to be created. NPCOL must be at least one.
   *
   * GRID    (local input/output)          HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information to be initialized.
   *
   * ---------------------------------------------------------------------
   */

  int hdim, hplerr = MPI_SUCCESS, ierr, ip2, k, mask, mycol, myrow, nprocs,
            rank, size;
  int local_myrow, local_mycol;

  MPI_Comm_rank(COMM, &rank);
  MPI_Comm_size(COMM, &size);
  /*
   * Abort if illegal process grid
   */
  nprocs = NPROW * NPCOL;
  if((nprocs > size) || (NPROW < 1) || (NPCOL < 1)) {
    HPL_pabort(__LINE__, "HPL_grid_init", "Illegal Grid");
  }
  /*
   * Row- or column-major ordering of the processes
   */
  int local_size = p * q;
  int local_rank = rank % local_size;
  int node       = rank / local_size; // node number

  if(ORDER == HPL_ROW_MAJOR) {
    GRID->order = HPL_ROW_MAJOR;
    local_mycol = local_rank % q;
    local_myrow = local_rank / q;

    int noderow = node / (NPCOL / q);
    int nodecol = node % (NPCOL / q);

    myrow = noderow * p + local_myrow;
    mycol = nodecol * q + local_mycol;

    myrow = rank / NPCOL;
    mycol = rank - myrow * NPCOL;
  } else {
    GRID->order = HPL_COLUMN_MAJOR;
    local_mycol = local_rank / p;
    local_myrow = local_rank % p;

    int noderow = node % (NPROW / p);
    int nodecol = node / (NPROW / p);

    myrow = noderow * p + local_myrow;
    mycol = nodecol * q + local_mycol;
  }

  GRID->iam         = rank;
  GRID->local_myrow = local_myrow;
  GRID->local_mycol = local_mycol;
  GRID->myrow       = myrow;
  GRID->mycol       = mycol;
  GRID->local_nprow = p;
  GRID->local_npcol = q;
  GRID->nprow       = NPROW;
  GRID->npcol       = NPCOL;
  GRID->nprocs      = nprocs;
  /*
   * row_ip2   : largest power of two <= nprow;
   * row_hdim  : row_ip2 procs hypercube dim;
   * row_ip2m1 : largest power of two <= nprow-1;
   * row_mask  : row_ip2m1 procs hypercube mask;
   */
  hdim = 0;
  ip2  = 1;
  k    = NPROW;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    hdim++;
  }
  GRID->row_ip2  = ip2;
  GRID->row_hdim = hdim;

  mask = ip2 = 1;
  k          = NPROW - 1;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    mask <<= 1;
    mask++;
  }
  GRID->row_ip2m1 = ip2;
  GRID->row_mask  = mask;
  /*
   * col_ip2   : largest power of two <= npcol;
   * col_hdim  : col_ip2 procs hypercube dim;
   * col_ip2m1 : largest power of two <= npcol-1;
   * col_mask  : col_ip2m1 procs hypercube mask;
   */
  hdim = 0;
  ip2  = 1;
  k    = NPCOL;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    hdim++;
  }
  GRID->col_ip2  = ip2;
  GRID->col_hdim = hdim;

  mask = ip2 = 1;
  k          = NPCOL - 1;
  while(k > 1) {
    k >>= 1;
    ip2 <<= 1;
    mask <<= 1;
    mask++;
  }
  GRID->col_ip2m1 = ip2;
  GRID->col_mask  = mask;
  /*
   * All communicator, leave if I am not part of this grid. Creation of the
   * row- and column communicators.
   */
  ierr = MPI_Comm_split(
      COMM, (rank < nprocs ? 0 : MPI_UNDEFINED), rank, &(GRID->all_comm));
  if(GRID->all_comm == MPI_COMM_NULL) return (ierr);

  ierr = MPI_Comm_split(GRID->all_comm, myrow, mycol, &(GRID->row_comm));
  if(ierr != MPI_SUCCESS) hplerr = ierr;

  ierr = MPI_Comm_split(GRID->all_comm, mycol, myrow, &(GRID->col_comm));
  if(ierr != MPI_SUCCESS) hplerr = ierr;

  return (hplerr);
}
