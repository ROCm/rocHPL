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
#ifndef HPL_COMM_HPP
#define HPL_COMM_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.hpp"
#include "hpl_panel.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPL_1RING   = 401, /* Unidirectional ring */
  HPL_1RING_M = 402, /* Unidirectional ring (modified) */
  HPL_2RING   = 403, /* Bidirectional ring */
  HPL_2RING_M = 404, /* Bidirectional ring (modified) */
  HPL_BLONG   = 405, /* long broadcast */
  HPL_BLONG_M = 406, /* long broadcast (modified) */
} HPL_T_TOP;

typedef MPI_Op HPL_T_OP;

#define HPL_SUM MPI_SUM
#define HPL_MAX MPI_MAX
#define HPL_MIN MPI_MIN

extern MPI_Op       HPL_DMXSWP;
extern MPI_Datatype PDFACT_ROW;
/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define HPL_FAILURE 0
#define HPL_SUCCESS 1
/*
 * ---------------------------------------------------------------------
 * comm function prototypes
 * ---------------------------------------------------------------------
 */
int HPL_send(double*, int, int, int, MPI_Comm);
int HPL_recv(double*, int, int, int, MPI_Comm);
int HPL_sdrv(double*, int, int, double*, int, int, int, MPI_Comm);
int HPL_bcast(double*, int, int, MPI_Comm, HPL_T_TOP top);
int HPL_bcast_1ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_bcast_1rinM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_bcast_2ring(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_bcast_2rinM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_bcast_blong(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_bcast_blonM(double* SBUF, int SCOUNT, int ROOT, MPI_Comm COMM);
int HPL_scatterv(double*, const int*, const int*, const int, int, MPI_Comm);
int HPL_allgatherv(double*, const int, const int*, const int*, MPI_Comm);
int HPL_barrier(MPI_Comm);
int HPL_broadcast(void*, const int, const HPL_T_TYPE, const int, MPI_Comm);

int HPL_reduce(void*,
               const int,
               const HPL_T_TYPE,
               const HPL_T_OP,
               const int,
               MPI_Comm);

int HPL_all_reduce(void*,
                   const int,
                   const HPL_T_TYPE,
                   const HPL_T_OP,
                   MPI_Comm);

void HPL_dmxswp(void*, void*, int*, MPI_Datatype*);
void HPL_all_reduce_dmxswp(double*, const int, const int, MPI_Comm, double*);

#endif
/*
 * End of hpl_comm.hpp
 */
