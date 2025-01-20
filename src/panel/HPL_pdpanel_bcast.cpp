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

int HPL_pdpanel_bcast(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanel_bcast broadcasts  the  current  panel.  Successful  completion
   * is indicated by a return code of HPL_SUCCESS.
   *
   * Arguments
   * =========
   *
   * PANEL   (input/output)                HPL_T_panel *
   *         On entry,  PANEL  points to the  current panel data structure
   *         being broadcast.
   *
   * ---------------------------------------------------------------------
   */

  if(PANEL == NULL) { return HPL_SUCCESS; }
  if(PANEL->grid->npcol <= 1) { return HPL_SUCCESS; }

  MPI_Comm comm = PANEL->grid->row_comm;
  int      root = PANEL->pcol;

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_LBCAST);
#endif
  /*
   * Single Bcast call
   */
  int err = HPL_bcast(PANEL->A0, PANEL->len, root, comm, PANEL->algo->btopo);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_LBCAST);
#endif

  return err;
}
