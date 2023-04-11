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
#ifndef HPL_PGESV_HPP
#define HPL_PGESV_HPP
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.hpp"
#include "hpl_blas.hpp"
#include "hpl_auxil.hpp"

#include "hpl_pmisc.hpp"
#include "hpl_grid.hpp"
#include "hpl_comm.hpp"
#include "hpl_pauxil.hpp"
#include "hpl_panel.hpp"
#include "hpl_pfact.hpp"

/*
 * ---------------------------------------------------------------------
 * #typedefs and data structures
 * ---------------------------------------------------------------------
 */
typedef enum {
  HPL_LEFT_LOOKING  = 301, /* Left looking lu fact variant */
  HPL_CROUT         = 302, /* Crout lu fact variant */
  HPL_RIGHT_LOOKING = 303  /* Right looking lu fact variant */
} HPL_T_FACT;

typedef enum {
  HPL_SWAP00 = 451, /* Use HPL_pdlaswp00 */
  HPL_SWAP01 = 452, /* Use HPL_pdlaswp01 */
  HPL_SW_MIX = 453, /* Use HPL_pdlaswp00_ for small number of */
                    /* columns, and HPL_pdlaswp01_ otherwise. */
  HPL_NO_SWP = 499
} HPL_T_SWAP;

typedef enum {
  HPL_LOOK_AHEAD = 0, /* look-ahead update */
  HPL_UPD_1      = 1, /* first update */
  HPL_UPD_2      = 2, /* second update */

  HPL_N_UPD = 3
} HPL_T_UPD;

typedef void (*HPL_T_UPD_FUN)(HPL_T_panel*, const HPL_T_UPD);

typedef struct HPL_S_palg {
  HPL_T_TOP     btopo; /* row broadcast topology */
  int           depth; /* look-ahead depth */
  int           nbdiv; /* recursive division factor */
  int           nbmin; /* recursion stopping criterium */
  HPL_T_FACT    pfact; /* panel fact variant */
  HPL_T_FACT    rfact; /* recursive fact variant */
  HPL_T_PFA_FUN pffun; /* panel fact function ptr */
  HPL_T_RFA_FUN rffun; /* recursive fact function ptr */
  HPL_T_UPD_FUN upfun; /* update function */
  HPL_T_SWAP    fswap; /* Swapping algorithm */
  int           fsthr; /* Swapping threshold */
  int           equil; /* Equilibration */
  int           align; /* data alignment constant */
  double        frac;  /* update split percentage */
} HPL_T_palg;

typedef struct HPL_S_pmat {
  double* A;   /* pointer to local piece of A */
  double* X;   /* pointer to solution vector */
  int     n;    /* global problem size */
  int     nb;   /* blocking factor */
  int     ld;   /* local leading dimension */
  int     mp;   /* local number of rows */
  int     nq;   /* local number of columns */
  int     info; /* computational flag */
  double* W;
} HPL_T_pmat;

extern hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
extern hipEvent_t swapUCopyEvent[HPL_N_UPD], swapWCopyEvent[HPL_N_UPD];
extern hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];

/*
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define MSGID_BEGIN_PFACT 1001 /* message id ranges */
#define MSGID_END_PFACT 2000
#define MSGID_BEGIN_FACT 2001
#define MSGID_END_FACT 3000
#define MSGID_BEGIN_PTRSV 3001
#define MSGID_END_PTRSV 4000

#define MSGID_BEGIN_COLL 9001
#define MSGID_END_COLL 10000
/*
 * ---------------------------------------------------------------------
 * #define macros definitions
 * ---------------------------------------------------------------------
 */
#define MNxtMgid(id_, beg_, end_) (((id_) + 1 > (end_) ? (beg_) : (id_) + 1))
/*
 * ---------------------------------------------------------------------
 * Function prototypes
 * ---------------------------------------------------------------------
 */

void HPL_pipid(HPL_T_panel*, int*, int*);
void HPL_piplen(HPL_T_panel*, const int, const int*, int*, int*);
void HPL_perm(const int, int*, int*, int*);

void HPL_plindx(HPL_T_panel*,
                const int,
                const int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*,
                int*);

void HPL_pdlaswp_start(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdlaswp_exchange(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdlaswp_end(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void HPL_pdupdateNT(HPL_T_panel*, const HPL_T_UPD);
void HPL_pdupdateTT(HPL_T_panel*, const HPL_T_UPD);
void HPL_pdgesv(HPL_T_grid*, HPL_T_palg*, HPL_T_pmat*);
void HPL_pdtrsv(HPL_T_grid*, HPL_T_pmat*);

#endif
/*
 * End of hpl_pgesv.hpp
 */
