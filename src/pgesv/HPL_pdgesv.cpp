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

void HPL_pdgesv(HPL_T_grid* GRID, HPL_T_palg* ALGO, HPL_T_pmat* A) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdgesv factors a N+1-by-N matrix using LU factorization with row
   * partial pivoting.  The main algorithm  is the "right looking" variant
   * with  or  without look-ahead.  The  lower  triangular  factor is left
   * unpivoted and the pivots are not returned. The right hand side is the
   * N+1 column of the coefficient matrix.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * ALGO    (global input)                HPL_T_palg *
   *         On entry,  ALGO  points to  the data structure containing the
   *         algorithmic parameters.
   *
   * A       (local input/output)          HPL_T_pmat *
   *         On entry, A points to the data structure containing the local
   *         array information.
   *
   * ---------------------------------------------------------------------
   */

  if(A->n <= 0) return;

  A->info = 0;

  HPL_T_panel * p, **panel = NULL;
  HPL_T_UPD_FUN HPL_pdupdate;
  int N, icurcol = 0, j, jb, jj = 0, jstart, k, mycol, n, nb, nn, npcol, nq,
         tag = MSGID_BEGIN_FACT, test;
#ifdef HPL_PROGRESS_REPORT
  double start_time, time, step_time, gflops, step_gflops;
#endif

  // depth        = ALGO->depth;
  const int depth = 1; // NC: Hardcoded now

  mycol        = GRID->mycol;
  npcol        = GRID->npcol;
  HPL_pdupdate = ALGO->upfun;
  N            = A->n;
  nb           = A->nb;

  if(N <= 0) return;

#ifdef HPL_PROGRESS_REPORT
  start_time = HPL_ptimer_walltime();
#endif

  /*
   * Allocate a panel list of length depth + 1 (depth >= 1)
   */
  panel = (HPL_T_panel**)malloc((size_t)(depth + 1) * sizeof(HPL_T_panel*));
  if(panel == NULL) {
    HPL_pabort(__LINE__, "HPL_pdgesvK2", "Memory allocation failed");
  }
  /*
   * Create and initialize the first panel
   */
  nq     = HPL_numroc(N + 1, nb, nb, mycol, 0, npcol);
  nn     = N;
  jstart = 0;

  jb = Mmin(nn, nb);
  HPL_pdpanel_new(
      GRID, ALGO, nn, nn + 1, jb, A, jstart, jstart, tag, &panel[0]);
  nn -= jb;
  jstart += jb;
  if(mycol == icurcol) {
    jj += jb;
    nq -= jb;
  }
  icurcol = MModAdd1(icurcol, npcol);
  tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Create second panel
   */
  HPL_pdpanel_new(
      GRID, ALGO, nn, nn + 1, Mmin(nn, nb), A, jstart, jstart, tag, &panel[1]);
  tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Initialize the lookahead - Factor jstart columns: panel[0]
   */
  jb = jstart;
  jb = Mmin(jb, nb);
  /*
   * Factor and broadcast 0-th panel
   */
  HPL_pdpanel_SendToHost(panel[0]);
  HPL_pdpanel_Wait(panel[0]);

  HPL_pdfact(panel[0]);

  // send the panel back to device before bcast
  HPL_pdpanel_SendToDevice(panel[0]);
  HPL_pdpanel_Wait(panel[0]);

  HPL_pdpanel_bcast(panel[0]);

  // start Ubcast+row swapping for second part of A
  HPL_pdlaswp_start(panel[0], HPL_UPD_2);

  if(mycol == icurcol) {
    // start Ubcast+row swapping for look ahead
    HPL_pdlaswp_start(panel[0], HPL_LOOK_AHEAD);
  }

  // start Ubcast+row swapping for first part of A
  HPL_pdlaswp_start(panel[0], HPL_UPD_1);

  // Ubcast+row swaps for second part of A
  HPL_pdlaswp_exchange(panel[0], HPL_UPD_2);

  if(mycol == icurcol) {
    // Ubcast+row swaps for look ahead
    // nn = HPL_numrocI(jb, j, nb, nb, mycol, 0, npcol);
    HPL_pdlaswp_exchange(panel[0], HPL_LOOK_AHEAD);
  }

  double stepStart, stepEnd;

#ifdef HPL_PROGRESS_REPORT
#ifdef HPL_DETAILED_TIMING
  float  smallDgemmTime, largeDgemm1Time, largeDgemm2Time;
  double smallDgemmGflops, largeDgemm1Gflops, largeDgemm2Gflops;

  if(GRID->myrow == 0 && mycol == 0) {
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "---------------------------------------------\n");
    printf("   %%   | Column    | Step Time (s) ||         DGEMM GFLOPS        "
           " || Panel Copy(s) | pdfact (s) | pmxswp (s) | Lbcast (s) | laswp "
           "(s) | GPU Sync (s) | Step GFLOPS | Overall GFLOPS\n");
    printf("       |           |               |  Small   |  First   | Second  "
           " |               |            |            |            |          "
           " |              |             |               \n");
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "---------------------------------------------\n");
  }
#else
  if(GRID->myrow == 0 && mycol == 0) {
    printf("---------------------------------------------------\n");
    printf("   %%   | Column    | Step Time (s) | Overall GFLOPS\n");
    printf("       |           |               |               \n");
    printf("---------------------------------------------------\n");
  }
#endif
#endif

  /*
   * Main loop over the remaining columns of A
   */
  for(j = jstart; j < N; j += nb) {
    HPL_ptimer_stepReset(HPL_TIMING_N, HPL_TIMING_BEG);

    stepStart = MPI_Wtime();
    n         = N - j;
    jb        = Mmin(n, nb);
    /*
     * Initialize current panel - Finish latest update, Factor and broadcast
     * current panel
     */
    (void)HPL_pdpanel_free(panel[1]);
    HPL_pdpanel_init(GRID, ALGO, n, n + 1, jb, A, j, j, tag, panel[1]);

    if(mycol == icurcol) {
      /* update look ahead */
      HPL_pdlaswp_end(panel[0], HPL_LOOK_AHEAD);
      HPL_pdupdate(panel[0], HPL_LOOK_AHEAD);

      // when the look ahead update is finished, copy back the current panel
      CHECK_HIP_ERROR(
          hipStreamWaitEvent(dataStream, update[HPL_LOOK_AHEAD], 0));
      HPL_pdpanel_SendToHost(panel[1]);

      /* Queue up finishing the second section */
      HPL_pdlaswp_end(panel[0], HPL_UPD_2);
      HPL_pdupdate(panel[0], HPL_UPD_2);

#ifdef HPL_DETAILED_TIMING
      HPL_ptimer(HPL_TIMING_UPDATE);
      CHECK_HIP_ERROR(hipEventSynchronize(update[HPL_LOOK_AHEAD]));
      HPL_ptimer(HPL_TIMING_UPDATE);
#endif

      // wait for the panel to arrive
      HPL_pdpanel_Wait(panel[0]);

#ifdef HPL_PROGRESS_REPORT
#ifdef HPL_DETAILED_TIMING
      const int curr = (panel[0]->grid->myrow == panel[0]->prow ? 1 : 0);
      const int mp   = panel[0]->mp - (curr != 0 ? jb : 0);

      // compute the GFLOPs of the look ahead update DGEMM
      CHECK_HIP_ERROR(hipEventElapsedTime(&smallDgemmTime,
                                          dgemmStart[HPL_LOOK_AHEAD],
                                          dgemmStop[HPL_LOOK_AHEAD]));
      smallDgemmGflops =
          (2.0 * mp * jb * jb) / (1000.0 * 1000.0 * smallDgemmTime);
#endif
#endif

      /*Panel factorization FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB +
       * (N-i*NB)(NB^2-NB)*/
      HPL_pdfact(panel[1]); /* factor current panel */

      // send the panel back to device before bcast
      HPL_pdpanel_SendToDevice(panel[1]);
      HPL_pdpanel_Wait(panel[0]);
    } else {
      /* Queue up finishing the second section */
      HPL_pdlaswp_end(panel[0], HPL_UPD_2);
      HPL_pdupdate(panel[0], HPL_UPD_2);
    }

    /* broadcast current panel */
    HPL_pdpanel_bcast(panel[1]);

    // start Ubcast+row swapping for second part of A
    HPL_pdlaswp_start(panel[1], HPL_UPD_2);

    // while the second section is updating, exchange the rows from the first
    // section
    HPL_pdlaswp_exchange(panel[0], HPL_UPD_1);

    /* Queue up finishing the first section */
    HPL_pdlaswp_end(panel[0], HPL_UPD_1);
    HPL_pdupdate(panel[0], HPL_UPD_1);

    if(mycol == icurcol) {
      jj += jb;
      nq -= jb;
    }
    icurcol = MModAdd1(icurcol, npcol);
    tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

    if(mycol == icurcol) {
      // prep the row swaps for the next look ahead
      //  nn = HPL_numrocI(jb, j+nb, nb, nb, mycol, 0, npcol);
      HPL_pdlaswp_start(panel[1], HPL_LOOK_AHEAD);

      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(panel[1], HPL_UPD_1);

      HPL_pdlaswp_exchange(panel[1], HPL_UPD_2);

      HPL_pdlaswp_exchange(panel[1], HPL_LOOK_AHEAD);
    } else {
      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(panel[1], HPL_UPD_1);

      HPL_pdlaswp_exchange(panel[1], HPL_UPD_2);
    }

    // wait here for the updates to compete
#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
#endif
    CHECK_HIP_ERROR(hipDeviceSynchronize());
#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_UPDATE);
#endif

    stepEnd = MPI_Wtime();

#ifdef HPL_PROGRESS_REPORT
#ifdef HPL_DETAILED_TIMING
    const int curr = (panel[0]->grid->myrow == panel[0]->prow ? 1 : 0);
    const int mp   = panel[0]->mp - (curr != 0 ? jb : 0);

    largeDgemm1Time = 0.0;
    largeDgemm2Time = 0.0;
    if(panel[0]->nu1) {
      CHECK_HIP_ERROR(hipEventElapsedTime(
          &largeDgemm1Time, dgemmStart[HPL_UPD_1], dgemmStop[HPL_UPD_1]));
      largeDgemm1Gflops = (2.0 * mp * jb * (panel[0]->nu1)) /
                          (1000.0 * 1000.0 * (largeDgemm1Time));
    }
    if(panel[0]->nu2) {
      CHECK_HIP_ERROR(hipEventElapsedTime(
          &largeDgemm2Time, dgemmStart[HPL_UPD_2], dgemmStop[HPL_UPD_2]));
      largeDgemm2Gflops = (2.0 * mp * jb * (panel[0]->nu2)) /
                          (1000.0 * 1000.0 * (largeDgemm2Time));
    }
#endif
    /* if this is process 0,0 and not the first panel */
    if(GRID->myrow == 0 && mycol == 0 && j > 0) {
      time      = HPL_ptimer_walltime() - start_time;
      step_time = stepEnd - stepStart;
      /*
      Step FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB
                          + 2*n*NB^2 - n*NB + 2*NB*n^2

      Overall FLOP count is (2/3)(N^3-n^3) - (1/2)(N^2-n^2) - (1/6)(N-n)
      */
      step_gflops =
          ((2.0 / 3.0) * jb * jb * jb - (1.0 / 2.0) * jb * jb -
           (1.0 / 6.0) * jb + 2.0 * n * jb * jb - jb * n + 2.0 * jb * n * n) /
          (step_time > 0.0 ? step_time : 1.e-6) / 1.e9;
      gflops = ((2.0 / 3.0) * (N * (double)N * N - n * (double)n * n) -
                (1.0 / 2.0) * (N * (double)N - n * (double)n) -
                (1.0 / 6.0) * ((double)N - (double)n)) /
               (time > 0.0 ? time : 1.e-6) / 1.e9;
      printf("%5.1f%% | %09d | ", j * 100.0 / N, j);
      printf("   %9.7f  |", stepEnd - stepStart);

#ifdef HPL_DETAILED_TIMING
      if(panel[0]->nu0) {
        printf(" %9.3e|", smallDgemmGflops);
      } else {
        printf("          |");
      }
      if(panel[0]->nu2) {
        printf(" %9.3e|", largeDgemm2Gflops);
      } else {
        printf("          |");
      }

      if(panel[0]->nu1) {
        printf(" %9.3e|", largeDgemm1Gflops);
      } else {
        printf("          |");
      }

      if(panel[0]->nu0) {
        printf("   %9.3e   |  %9.3e |  %9.3e |",
               HPL_ptimer_getStep(HPL_TIMING_COPY),
               HPL_ptimer_getStep(HPL_TIMING_RPFACT),
               HPL_ptimer_getStep(HPL_TIMING_MXSWP));
      } else {
        printf("               |            |            |");
      }

      printf("  %9.3e | %9.3e |   %9.3e  |",
             HPL_ptimer_getStep(HPL_TIMING_LBCAST),
             HPL_ptimer_getStep(HPL_TIMING_LASWP),
             HPL_ptimer_getStep(HPL_TIMING_UPDATE));

      printf("  %9.3e  |", step_gflops);
#endif

      printf("    %9.3e   \n", gflops);
    }
#endif

    /*
     * Circular  of the panel pointers:
     * xtmp = x[0]; for( k=0; k < 1; k++ ) x[k] = x[k+1]; x[d] = xtmp;
     *
     * Go to next process row and column - update the message ids for broadcast
     */
    p        = panel[0];
    panel[0] = panel[1];
    panel[1] = p;
  }
  /*
   * Clean-up: Finish updates - release panels and panel list
   */
  // nn = HPL_numrocI(1, N, nb, nb, mycol, 0, npcol);
  HPL_pdlaswp_end(panel[0], HPL_LOOK_AHEAD);
  HPL_pdupdate(panel[0], HPL_LOOK_AHEAD);

  HPL_pdlaswp_end(panel[0], HPL_UPD_2);
  HPL_pdupdate(panel[0], HPL_UPD_2);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
  CHECK_HIP_ERROR(hipDeviceSynchronize());
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif

  HPL_pdpanel_disp(&panel[0]);
  HPL_pdpanel_disp(&panel[1]);
  if(panel) free(panel);

  /*
   * Solve upper triangular system
   */
  if(A->info == 0) HPL_pdtrsv(GRID, A);
}
