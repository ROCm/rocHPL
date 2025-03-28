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
#ifndef HPL_PANEL_HPP
#define HPL_PANEL_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"

/*
 * ---------------------------------------------------------------------
 * Data Structures
 * ---------------------------------------------------------------------
 */
typedef struct HPL_S_panel {
  struct HPL_S_grid* grid;  /* ptr to the process grid */
  struct HPL_S_palg* algo;  /* ptr to the algo parameters */
  struct HPL_S_pmat* pmat;  /* ptr to the local array info */
  double*            A;     /* ptr to trailing part of A */
  double*            A0;    /* ptr to current panel of A */
  double*            L2;    /* ptr to L */
  double*            L1;    /* ptr to jb x jb upper block of A */
  double*            U0;    /* ptr to U */
  double*            U1;    /* ptr to U1 */
  double*            U2;    /* ptr to U2 */
  double*            timers;  /* timers work space */
  int*               IWORK; /* integer workspace for swapping */
  int*               ipiv;
  int*               dipiv;
  int                nu0;
  int                nu1;
  int                nu2;
  int                ldu0;
  int                ldu1;
  int                ldu2;
  int                lda0;       /* local leading dim of array A0 */
  int                ldl2;       /* local leading dim of array L2 */
  int                len;        /* length of the buffer to broadcast */
  void*              buffers[2]; /* buffers for panel bcast */
  int                counts[2];  /* counts for panel bcast */
  MPI_Datatype       dtypes[2];  /* data types for panel bcast */
  MPI_Request        request[1]; /* requests for panel bcast */
  MPI_Status         status[1];  /* status for panel bcast */
  int                nb;         /* distribution blocking factor */
  int                jb;         /* panel width */
  int                m;          /* global # of rows of trailing part of A */
  int                n;          /* global # of cols of trailing part of A */
  int                ia;         /* global row index of trailing part of A */
  int                ja;         /* global col index of trailing part of A */
  int                mp;         /* local # of rows of trailing part of A */
  int                nq;         /* local # of cols of trailing part of A */
  int                ii;         /* local row index of trailing part of A */
  int                jj;         /* local col index of trailing part of A */
  int                lda;        /* local leading dim of array A */
  int                prow;       /* proc. row owning 1st row of trail. A */
  int                pcol;       /* proc. col owning 1st col of trail. A */
  int                msgid;      /* message id for panel bcast */
} HPL_T_panel;

/*
 * ---------------------------------------------------------------------
 * panel function prototypes
 * ---------------------------------------------------------------------
 */
#include "hpl_pgesv.hpp"

typedef struct HPL_S_test HPL_T_test;

int HPL_pdpanel_new(HPL_T_test*,
                    HPL_T_grid*,
                    HPL_T_palg*,
                    HPL_T_pmat*,
                    HPL_T_panel*,
                    size_t&);

void HPL_pdpanel_init(HPL_T_grid*,
                      HPL_T_palg*,
                      const int,
                      const int,
                      const int,
                      HPL_T_pmat*,
                      const int,
                      const int,
                      const int,
                      HPL_T_panel*);

int  HPL_pdpanel_free(HPL_T_panel*);
void HPL_pdpanel_SendToHost(HPL_T_panel*);
void HPL_pdpanel_SendToDevice(HPL_T_panel*);
void HPL_pdpanel_swapids(HPL_T_panel* PANEL);
void HPL_pdpanel_copyL1(HPL_T_panel* PANEL);
void HPL_pdpanel_Wait(HPL_T_panel* PANEL);
int  HPL_pdpanel_bcast(HPL_T_panel*);
#endif
/*
 * End of hpl_panel.hpp
 */
