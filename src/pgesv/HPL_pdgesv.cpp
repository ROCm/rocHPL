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

  HPL_T_UPD_FUN HPL_pdupdate;
  int N, icurcol = 0, j, jb, jj = 0, jstart, k, mycol, myrow, n, nb, nn, npcol, nq,
         tag = MSGID_BEGIN_FACT;
#ifdef HPL_PROGRESS_REPORT
  double start_time, time, step_time, gflops, step_gflops;
#endif

  myrow        = GRID->myrow;
  mycol        = GRID->mycol;
  npcol        = GRID->npcol;
  HPL_pdupdate = ALGO->upfun;
  N            = A->n;
  nb           = A->nb;

  // swapping workspaces
  double *W0 = A->W0;
  double *W1 = A->W1;
  double *W2 = A->W2;

  if(N <= 0) return;

#ifdef HPL_PROGRESS_REPORT
  start_time = HPL_ptimer_walltime();
#endif

  HPL_T_panel* curr = &(A->panel[0]);
  HPL_T_panel* next = &(A->panel[1]);

  /*
   * initialize the first panel
   */
  nq     = HPL_numroc(N + 1, nb, nb, mycol, 0, npcol);
  nn     = N;
  jstart = 0;

  jb = Mmin(nn, nb);
  HPL_pdpanel_init(
      GRID, ALGO, nn, nn + 1, jb, A, jstart, jstart, tag, curr);
  nn -= jb;
  jstart += jb;
  if(mycol == icurcol) {
    jj += jb;
    nq -= jb;
  }
  icurcol = MModAdd1(icurcol, npcol);
  tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * initialize second panel
   */
  HPL_pdpanel_init(
      GRID, ALGO, nn, nn + 1, Mmin(nn, nb), A, jstart, jstart, tag, next);
  tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

  /*
   * Initialize the lookahead - Factor jstart columns: panel[0]
   */
  jb = jstart;
  jb = Mmin(jb, nb);
  /*
   * Factor and broadcast 0-th panel
   */
  if (mycol == 0) {
    HPL_dlacpy_gpu(curr->mp,
                   curr->jb,
                   curr->A,
                   curr->lda,
                   curr->A0,
                   curr->lda0);

    HPL_pdfact(curr);

    CHECK_HIP_ERROR(hipEventSynchronize(pfactStop));

    if (myrow == curr->prow) {
      HPL_dlatcpy_gpu(curr->jb,
                      curr->jb,
                      curr->L1,
                      curr->jb,
                      Mptr(curr->A, 0, -curr->jb, curr->lda),
                      curr->lda);
    }

  }

  HPL_pdpanel_bcast(curr);

  // compute swapping info
  HPL_pdpanel_swapids(curr);

  // start Ubcast+row swapping for second part of A
  HPL_pdlaswp_start(curr,
                    curr->nu2,
                    curr->U2,
                    curr->ldu2,
                    W2,
                    curr->ldu2,
                    Mptr(curr->A, 0, curr->nu0 + curr->nu1, curr->lda),
                    curr->lda,
                    swapStartEvent[HPL_UPD_2]);

  if(mycol == icurcol) {
    // start Ubcast+row swapping for look ahead
    HPL_pdlaswp_start(curr,
                      curr->nu0,
                      curr->U0,
                      curr->ldu0,
                      W0,
                      curr->ldu0,
                      curr->A,
                      curr->lda,
                      swapStartEvent[HPL_LOOK_AHEAD]);
  }

  // start Ubcast+row swapping for first part of A
  HPL_pdlaswp_start(curr,
                    curr->nu1,
                    curr->U1,
                    curr->ldu1,
                    W1,
                    curr->ldu1,
                    Mptr(curr->A, 0, curr->nu0, curr->lda),
                    curr->lda,
                    swapStartEvent[HPL_UPD_1]);

  // Ubcast+row swaps for second part of A
  HPL_pdlaswp_exchange(curr,
                       curr->nu2,
                       curr->U2,
                       curr->ldu2,
                       W2,
                       curr->ldu2,
                       Mptr(curr->A, 0, curr->nu0 + curr->nu1, curr->lda),
                       curr->lda,
                       swapStartEvent[HPL_UPD_2]);

  if(mycol == icurcol) {
    // Ubcast+row swaps for look ahead
    HPL_pdlaswp_exchange(curr,
                         curr->nu0,
                         curr->U0,
                         curr->ldu0,
                         W0,
                         curr->ldu0,
                         curr->A,
                         curr->lda,
                         swapStartEvent[HPL_LOOK_AHEAD]);
  }

  double stepStart, stepEnd;

#ifdef HPL_PROGRESS_REPORT
#ifdef HPL_DETAILED_TIMING
  float  smallDgemmTime, largeDgemm1Time, largeDgemm2Time;
  double smallDgemmGflops, largeDgemm1Gflops, largeDgemm2Gflops;

  if(GRID->myrow == 0 && mycol == 0) {
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
    printf("   %%   | Column    | Step Time (s) ||         DGEMM GFLOPS        "
           " || pdfact (s) | pmxswp (s) | Lbcast (s) | laswp "
           "(s) | GPU Sync (s) | Step GFLOPS | Overall GFLOPS\n");
    printf("       |           |               |  Small   |  First   | Second  "
           " |            |            |            |          "
           " |              |             |               \n");
    printf("-------------------------------------------------------------------"
           "-------------------------------------------------------------------"
           "------------------------------\n");
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
    HPL_pdpanel_init(GRID, ALGO, n, n + 1, jb, A, j, j, tag, next);

    if(mycol == icurcol) {
      /* update look ahead */
      HPL_pdlaswp_end(curr,
                      curr->nu0,
                      curr->U0,
                      curr->ldu0,
                      W0,
                      curr->ldu0,
                      curr->A,
                      curr->lda);

      HPL_pdupdate(curr, HPL_LOOK_AHEAD);

      HPL_dlacpy_gpu(next->mp,
                     next->jb,
                     next->A,
                     next->lda,
                     next->A0,
                     next->lda0);

      /*Panel factorization FLOP count is (2/3)NB^3 - (1/2)NB^2 - (1/6)NB +
       * (N-i*NB)(NB^2-NB)*/
      HPL_pdfact(next); /* factor current panel */

      if (myrow == next->prow) {
        HPL_dlatcpy_gpu(next->jb,
                        next->jb,
                        next->L1,
                        next->jb,
                        Mptr(next->A, 0, -next->jb, next->lda),
                        next->lda);
      }
    }

    /* Queue up finishing the second section */
    HPL_pdlaswp_end(curr,
                    curr->nu2,
                    curr->U2,
                    curr->ldu2,
                    W2,
                    curr->ldu2,
                    Mptr(curr->A, 0, curr->nu0 + curr->nu1, curr->lda),
                    curr->lda);

    HPL_pdupdate(curr, HPL_UPD_2);

    if(mycol == icurcol) {
      CHECK_HIP_ERROR(hipEventSynchronize(pfactStop));
    }

    /* broadcast current panel */
    HPL_pdpanel_bcast(next);

    // compute swapping info
    HPL_pdpanel_swapids(next);

    // start Ubcast+row swapping for second part of A
    HPL_pdlaswp_start(next,
                      next->nu2,
                      next->U2,
                      next->ldu2,
                      W2,
                      next->ldu2,
                      Mptr(next->A, 0, next->nu0 + next->nu1, next->lda),
                      next->lda,
                      swapStartEvent[HPL_UPD_2]);

    // while the second section is updating, exchange the rows from the first
    // section
    HPL_pdlaswp_exchange(curr,
                         curr->nu1,
                         curr->U1,
                         curr->ldu1,
                         W1,
                         curr->ldu1,
                         Mptr(curr->A, 0, curr->nu0, curr->lda),
                         curr->lda,
                         swapStartEvent[HPL_UPD_1]);

    /* Queue up finishing the first section */
    HPL_pdlaswp_end(curr,
                    curr->nu1,
                    curr->U1,
                    curr->ldu1,
                    W1,
                    curr->ldu1,
                    Mptr(curr->A, 0, curr->nu0, curr->lda),
                    curr->lda);
    HPL_pdupdate(curr, HPL_UPD_1);

    if(mycol == icurcol) {
      jj += jb;
      nq -= jb;
    }
    icurcol = MModAdd1(icurcol, npcol);
    tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT);

    if(mycol == icurcol) {
      // prep the row swaps for the next look ahead
      //  nn = HPL_numrocI(jb, j+nb, nb, nb, mycol, 0, npcol);
      HPL_pdlaswp_start(next,
                        next->nu0,
                        next->U0,
                        next->ldu0,
                        W0,
                        next->ldu0,
                        next->A,
                        next->lda,
                        swapStartEvent[HPL_LOOK_AHEAD]);

      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(next,
                        next->nu1,
                        next->U1,
                        next->ldu1,
                        W1,
                        next->ldu1,
                        Mptr(next->A, 0, next->nu0, next->lda),
                        next->lda,
                        swapStartEvent[HPL_UPD_1]);

      HPL_pdlaswp_exchange(next,
                           next->nu2,
                           next->U2,
                           next->ldu2,
                           W2,
                           next->ldu2,
                           Mptr(next->A, 0, next->nu0 + next->nu1, next->lda),
                           next->lda,
                           swapStartEvent[HPL_UPD_2]);

      HPL_pdlaswp_exchange(next,
                           next->nu0,
                           next->U0,
                           next->ldu0,
                           W0,
                           next->ldu0,
                           next->A,
                           next->lda,
                           swapStartEvent[HPL_LOOK_AHEAD]);
    } else {
      // start Ubcast+row swapping for first part of A
      HPL_pdlaswp_start(next,
                        next->nu1,
                        next->U1,
                        next->ldu1,
                        W1,
                        next->ldu1,
                        Mptr(next->A, 0, next->nu0, next->lda),
                        next->lda,
                        swapStartEvent[HPL_UPD_1]);

      HPL_pdlaswp_exchange(next,
                           next->nu2,
                           next->U2,
                           next->ldu2,
                           W2,
                           next->ldu2,
                           Mptr(next->A, 0, next->nu0 + next->nu1, next->lda),
                           next->lda,
                           swapStartEvent[HPL_UPD_2]);
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
    const int icurr = (curr->grid->myrow == curr->prow ? 1 : 0);
    const int mp   = curr->mp - (icurr != 0 ? jb : 0);

    if(curr->nu0) {
      // compute the GFLOPs of the look ahead update DGEMM
      CHECK_HIP_ERROR(hipEventElapsedTime(&smallDgemmTime,
                                          dgemmStart[HPL_LOOK_AHEAD],
                                          dgemmStop[HPL_LOOK_AHEAD]));
      smallDgemmGflops =
          (2.0 * mp * jb * jb) / (1000.0 * 1000.0 * smallDgemmTime);
    }

    largeDgemm1Time = 0.0;
    largeDgemm2Time = 0.0;
    if(curr->nu1) {
      CHECK_HIP_ERROR(hipEventElapsedTime(
          &largeDgemm1Time, dgemmStart[HPL_UPD_1], dgemmStop[HPL_UPD_1]));
      largeDgemm1Gflops = (2.0 * mp * jb * (curr->nu1)) /
                          (1000.0 * 1000.0 * (largeDgemm1Time));
    }
    if(curr->nu2) {
      CHECK_HIP_ERROR(hipEventElapsedTime(
          &largeDgemm2Time, dgemmStart[HPL_UPD_2], dgemmStop[HPL_UPD_2]));
      largeDgemm2Gflops = (2.0 * mp * jb * (curr->nu2)) /
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
      if(curr->nu0) {
        printf(" %9.3e|", smallDgemmGflops);
      } else {
        printf("          |");
      }
      if(curr->nu2) {
        printf(" %9.3e|", largeDgemm2Gflops);
      } else {
        printf("          |");
      }

      if(curr->nu1) {
        printf(" %9.3e|", largeDgemm1Gflops);
      } else {
        printf("          |");
      }

      if(curr->nu0) {
        float pfactTime = 0.;
        CHECK_HIP_ERROR(hipEventElapsedTime(&pfactTime, pfactStart, pfactStop));

        printf("  %9.3e |  %9.3e |",
               static_cast<double>(pfactTime)/1000,
               HPL_ptimer_getStep(HPL_TIMING_MXSWP));
      } else {
        printf("            |            |");
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

    std::swap(curr,next);
  }

  /*
   * Clean-up: Finish updates - release panels and panel list
   */
  // nn = HPL_numrocI(1, N, nb, nb, mycol, 0, npcol);
  HPL_pdlaswp_end(curr,
                  curr->nu0,
                  curr->U0,
                  curr->ldu0,
                  W0,
                  curr->ldu0,
                  curr->A,
                  curr->lda);
  HPL_pdupdate(curr, HPL_LOOK_AHEAD);

  HPL_pdlaswp_end(curr,
                  curr->nu2,
                  curr->U2,
                  curr->ldu2,
                  W2,
                  curr->ldu2,
                  Mptr(curr->A, 0, curr->nu0 + curr->nu1, curr->lda),
                  curr->lda);
  HPL_pdupdate(curr, HPL_UPD_2);

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif
  CHECK_HIP_ERROR(hipDeviceSynchronize());
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_UPDATE);
#endif

  /*
   * Solve upper triangular system
   */
  if(A->info == 0) HPL_pdtrsv(GRID, A);
}
