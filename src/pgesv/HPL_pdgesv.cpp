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
#include <limits>


void print_pdfact_stats(HPL_T_panel* PANEL);
void print_bcast_stats(HPL_T_panel* PANEL);
void print_update_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void print_rowgather_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void print_rowexchange_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD);
void print_rowscatter_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD);

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

  // Reset timers
  pdfact_time=0.;
  bcast_time=0.;
  scatter_time[HPL_LOOK_AHEAD]=0.;
  scatter_time[HPL_UPD_1]=0.;
  scatter_time[HPL_UPD_2]=0.;
  gather_time[HPL_LOOK_AHEAD]=0.;
  gather_time[HPL_UPD_1]=0.;
  gather_time[HPL_UPD_2]=0.;
  for (int i=0;i<nb;++i) panel[0]->timers[i]=0.;
  for (int i=0;i<nb;++i) panel[1]->timers[i]=0.;

  MPI_Barrier(MPI_COMM_WORLD);

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

  CHECK_HIP_ERROR(hipDeviceSynchronize());



  if(GRID->myrow == 0 && mycol == 0) {
    printf("-----------------------------------\n");
    printf("Iteration: %5.1f%%, Column: %09d \n", 0.0, 0);
    printf("-----------------------------------\n");
  }

  print_pdfact_stats(panel[0]);
  print_bcast_stats(panel[0]);
  print_rowgather_stats(panel[0], HPL_UPD_2);
  print_rowgather_stats(panel[0], HPL_LOOK_AHEAD);
  print_rowgather_stats(panel[0], HPL_UPD_1);

  print_rowexchange_stats(panel[0], HPL_UPD_2);
  print_rowexchange_stats(panel[0], HPL_LOOK_AHEAD);


  double stepStart, stepEnd;

  /*
   * Main loop over the remaining columns of A
   */
  for(j = jstart; j < N; j += nb) {
    HPL_ptimer_stepReset(HPL_TIMING_N, HPL_TIMING_BEG);

    // Reset timers
    pdfact_time=0.;
    bcast_time=0.;
    scatter_time[HPL_LOOK_AHEAD]=0.;
    scatter_time[HPL_UPD_1]=0.;
    scatter_time[HPL_UPD_2]=0.;
    gather_time[HPL_LOOK_AHEAD]=0.;
    gather_time[HPL_UPD_1]=0.;
    gather_time[HPL_UPD_2]=0.;
    for (int i=0;i<nb;++i) panel[0]->timers[i]=0.;
    for (int i=0;i<nb;++i) panel[1]->timers[i]=0.;

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

    if(GRID->myrow == 0 && mycol == 0) {
      printf("-----------------------------------\n");
      printf("Iteration: %5.1f%%, Column: %09d \n", j * 100.0 / N, j);
      printf("-----------------------------------\n");
    }

    print_rowscatter_stats(panel[0], HPL_LOOK_AHEAD);
    print_update_stats(panel[0], HPL_LOOK_AHEAD);

    print_pdfact_stats(panel[1]);
    print_bcast_stats(panel[1]);
    print_rowexchange_stats(panel[0], HPL_UPD_1);

    print_rowscatter_stats(panel[0], HPL_UPD_2);
    print_update_stats(panel[0], HPL_UPD_2);
    print_rowgather_stats(panel[1], HPL_UPD_2);

    print_rowscatter_stats(panel[0], HPL_UPD_1);
    print_update_stats(panel[0], HPL_UPD_1);

    print_rowgather_stats(panel[1], HPL_LOOK_AHEAD);
    print_rowgather_stats(panel[1], HPL_UPD_1);

    print_rowexchange_stats(panel[1], HPL_UPD_2);
    print_rowexchange_stats(panel[1], HPL_LOOK_AHEAD);


    if(GRID->myrow == 0 && mycol == 0) {
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

      printf("Step GLFOPS =  %9.3e, Overall GFLOPS = %9.3e   \n\n", step_gflops, gflops);
    }
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

  if(GRID->myrow == 0 && mycol == 0) {
    printf("-----------------------------------\n");
    printf("Iteration: %5.1f%%, Column: %09d \n", 100.0, N);
    printf("-----------------------------------\n");
  }

  print_rowscatter_stats(panel[0], HPL_LOOK_AHEAD);
  print_update_stats(panel[0], HPL_LOOK_AHEAD);

  print_rowscatter_stats(panel[0], HPL_UPD_2);
  print_update_stats(panel[0], HPL_UPD_2);

  HPL_pdpanel_disp(&panel[0]);
  HPL_pdpanel_disp(&panel[1]);
  if(panel) free(panel);

  /*
   * Solve upper triangular system
   */
  if(A->info == 0) HPL_pdtrsv(GRID, A);
}

void print_pdfact_stats(HPL_T_panel* PANEL) {
  // Collect up pdfact timers

  int jb = PANEL->jb;

  // Compute stats on swaps
  // We assume the row swaps are essentially synchronous on each process in the MPI column,
  // so just use the timers that row==0 has, and send the stats to rank (0,0)

  double swap_avg=0.;
  double swap_min=std::numeric_limits<double>::max();
  double swap_max=std::numeric_limits<double>::min();

  for (int i=0;i<jb;++i) {
    swap_avg += PANEL->timers[i];
    swap_min = std::min(swap_min, PANEL->timers[i]);
    swap_max = std::max(swap_max, PANEL->timers[i]);
  }
  swap_avg /= jb;

  double swap_stddev=0.0;
  for (int i=0;i<jb;++i) {
    swap_stddev += (PANEL->timers[i]-swap_avg)*(PANEL->timers[i]-swap_avg);
  }
  swap_stddev /= jb;
  swap_stddev = sqrt(swap_stddev);

  //Get the size of the largest local matrix
  int mp_max=0;
  MPI_Reduce(&PANEL->mp, &mp_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

  //Get the largest pdfact timer in the column
  double pdfact_time_max=0;
  MPI_Reduce(&pdfact_time, &pdfact_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);

  //For swapping timers, noise can shift things around, but to isolate the MPI time only, we min across the column
  double swap_avg_col=0;
  double swap_min_col=0;
  double swap_max_col=0;
  double swap_stddev_col=0;
  MPI_Reduce(&swap_avg, &swap_avg_col, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
  MPI_Reduce(&swap_min, &swap_min_col, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
  MPI_Reduce(&swap_max, &swap_max_col, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
  MPI_Reduce(&swap_stddev, &swap_stddev_col, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);

  MPI_Request request[5];
  double pdfactTimeRoot=0.;
  double swapAvgRoot=0.;
  double swapMinRoot=0.;
  double swapMaxRoot=0.;
  double swapStdDevRoot=0.;

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Irecv(&pdfactTimeRoot, 1, MPI_DOUBLE, PANEL->pcol, 0, PANEL->grid->row_comm, request+0);
    MPI_Irecv(&swapAvgRoot,    1, MPI_DOUBLE, PANEL->pcol, 1, PANEL->grid->row_comm, request+1);
    MPI_Irecv(&swapMinRoot,    1, MPI_DOUBLE, PANEL->pcol, 2, PANEL->grid->row_comm, request+2);
    MPI_Irecv(&swapMaxRoot,    1, MPI_DOUBLE, PANEL->pcol, 3, PANEL->grid->row_comm, request+3);
    MPI_Irecv(&swapStdDevRoot, 1, MPI_DOUBLE, PANEL->pcol, 4, PANEL->grid->row_comm, request+4);
  }
  if (PANEL->grid->mycol==PANEL->pcol && PANEL->grid->myrow==0) {
    MPI_Send(&pdfact_time_max, 1, MPI_DOUBLE, 0, 0, PANEL->grid->row_comm);
    MPI_Send(&swap_avg_col,    1, MPI_DOUBLE, 0, 1, PANEL->grid->row_comm);
    MPI_Send(&swap_min_col,    1, MPI_DOUBLE, 0, 2, PANEL->grid->row_comm);
    MPI_Send(&swap_max_col,    1, MPI_DOUBLE, 0, 3, PANEL->grid->row_comm);
    MPI_Send(&swap_stddev_col, 1, MPI_DOUBLE, 0, 4, PANEL->grid->row_comm);
  }

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    MPI_Waitall(5, request, MPI_STATUSES_IGNORE);

    printf("Pdfact:                 MaxSize (%6d x %6d), Column %3d, Time (ms) = %8.3f,  Row Swaps Time (us) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), \n",
            mp_max, jb, PANEL->pcol, pdfactTimeRoot, swapMinRoot, swapAvgRoot, swapStdDevRoot, swapMaxRoot);
  }
}


void print_bcast_stats(HPL_T_panel* PANEL) {

  if (PANEL->grid->npcol==1) return;

  // Total time for this row's bcast is the max over the MPI row
  double bcast_time_total=0.;
  MPI_Reduce(&bcast_time, &bcast_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);

  // Min max avg over all rows' bcast
  double bcast_time_avg=0.;
  double bcast_time_min=0.;
  double bcast_time_max=0.;
  double bcast_time_stddev=0.;

  MPI_Allreduce(&bcast_time_total, &bcast_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
  MPI_Reduce(&bcast_time_total, &bcast_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
  MPI_Reduce(&bcast_time_total, &bcast_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
  bcast_time_avg /= PANEL->grid->nprow;

  double bcast_var = (bcast_time_total - bcast_time_avg)*(bcast_time_total - bcast_time_avg);
  MPI_Reduce(&bcast_var, &bcast_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
  bcast_time_stddev /= PANEL->grid->nprow;
  bcast_time_stddev = sqrt(bcast_time_stddev);

  int mp_max=0;
  MPI_Reduce(&PANEL->mp, &mp_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

  if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
    printf("LBcast:                 MaxSize (%6d x %6d), RootCol %2d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
            mp_max, PANEL->jb, PANEL->pcol, bcast_time_min, bcast_time_avg, bcast_time_stddev, bcast_time_max,
            mp_max*PANEL->jb*8/(1.0E6*bcast_time_avg));
  }
}

void print_rowgather_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {

  if (PANEL->grid->nprow==1) return;


  int n=0;
  std::string name;
  if(UPD == HPL_LOOK_AHEAD) {
    n   = PANEL->nu0;
    name = "Look Ahead";
  } else if(UPD == HPL_UPD_1) {
    n   = PANEL->nu1;
    name = "1";
  } else if(UPD == HPL_UPD_2) {
    n   = PANEL->nu2;
    name = "2";
  }

  double rowgather_time_avg=0.;
  double rowgather_time_min=0.;
  double rowgather_time_max=0.;
  double rowgather_time_stddev=0.;

  if (UPD == HPL_LOOK_AHEAD) {
    // Get time from GPU
    float rowgatherTime=0.;
    if (PANEL->grid->mycol==MModAdd1(PANEL->pcol, PANEL->grid->npcol)) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&rowgatherTime,
                                      rowGatherStart[UPD],
                                      rowGatherStop[UPD]));
    }

    double rowgather_time = rowgatherTime;

    // min max avg over the column
    MPI_Allreduce(&rowgather_time, &rowgather_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
    MPI_Reduce(&rowgather_time, &rowgather_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
    MPI_Reduce(&rowgather_time, &rowgather_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
    rowgather_time_avg /= PANEL->grid->nprow;

    double rowgather_var = (rowgather_time - rowgather_time_avg)*(rowgather_time - rowgather_time_avg);
    MPI_Reduce(&rowgather_var, &rowgather_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
    rowgather_time_stddev /= PANEL->grid->nprow;
    rowgather_time_stddev = sqrt(rowgather_time_stddev);


    double rowgather_time_avg_root=0.;
    double rowgather_time_min_root=0.;
    double rowgather_time_max_root=0.;
    double rowgather_time_stddev_root=0.;
    int n_root = 0;

    // Only one column should have done the lookahead, so just max over the row
    MPI_Reduce(&rowgather_time_avg, &rowgather_time_avg_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowgather_time_min, &rowgather_time_min_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowgather_time_max, &rowgather_time_max_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowgather_time_stddev, &rowgather_time_stddev_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&n, &n_root, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_root>0) {
      printf("RowGather Lookahead:    MaxSize (%6d x %6d), Column %3d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              PANEL->jb, n_root, MModAdd1(PANEL->pcol, PANEL->grid->npcol), rowgather_time_min_root, rowgather_time_avg_root, rowgather_time_stddev_root, rowgather_time_max_root,
              2*n_root*PANEL->jb*8/(1.0E6*rowgather_time_avg_root*PANEL->grid->nprow));
    }
  } else {
    // Get time from GPU
    float rowgatherTime=0.;
    CHECK_HIP_ERROR(hipEventElapsedTime(&rowgatherTime,
                                        rowGatherStart[UPD],
                                        rowGatherStop[UPD]));

    double rowgather_time = rowgatherTime;


    int n_max=0;
    MPI_Reduce(&n, &n_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    // Min max avg over all ranks
    MPI_Allreduce(&rowgather_time, &rowgather_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->all_comm);
    MPI_Reduce(&rowgather_time, &rowgather_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->all_comm);
    MPI_Reduce(&rowgather_time, &rowgather_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->all_comm);
    rowgather_time_avg /= PANEL->grid->nprocs;

    double rowgather_var = (rowgather_time - rowgather_time_avg)*(rowgather_time - rowgather_time_avg);
    MPI_Reduce(&rowgather_var, &rowgather_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->all_comm);
    rowgather_time_stddev /= PANEL->grid->nprocs;
    rowgather_time_stddev = sqrt(rowgather_time_stddev);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_max>0) {
      printf("RowGather %s:            MaxSize (%6d x %6d),             Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              name.c_str(), PANEL->jb, n_max, rowgather_time_min, rowgather_time_avg, rowgather_time_stddev, rowgather_time_max,
              2*n_max*PANEL->jb*8/(1.0E6*rowgather_time_avg*PANEL->grid->nprow));
    }
  }
}


void print_rowexchange_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {

  if (PANEL->grid->nprow==1) return;

  int n=0;
  std::string name;
  if(UPD == HPL_LOOK_AHEAD) {
    n   = PANEL->nu0;
    name = "Look Ahead";
  } else if(UPD == HPL_UPD_1) {
    n   = PANEL->nu1;
    name = "1";
  } else if(UPD == HPL_UPD_2) {
    n   = PANEL->nu2;
    name = "2";
  }

  if (UPD == HPL_LOOK_AHEAD) {
    double scatterTimeRoot=0.;
    double gatherTimeRoot=0.;
    int nroot = 0;

    // Total time for this col's scaller is the max over the col
    double scatter_time_total=0.;
    MPI_Reduce(&scatter_time[UPD], &scatter_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);

    // Total time for this col's scaller is the max over the col
    double gather_time_total=0.;
    MPI_Reduce(&gather_time[UPD], &gather_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);

    // Only one column should have done the lookahead, so just max over the row
    MPI_Reduce(&scatter_time_total, &scatterTimeRoot, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gather_time_total, &gatherTimeRoot, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&n, &nroot, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
      if (scatterTimeRoot>0.0) {
        printf("Scatterv Lookahead:     MaxSize (%6d x %6d), Column %3d, Time (ms) = %8.3f, BW Est (GB/s) = %6.2f\n",
                nroot, nroot,
                MModAdd1(PANEL->pcol, PANEL->grid->npcol),
                scatterTimeRoot,
                nroot*nroot*8/(1.0E6*scatterTimeRoot));
      }
      if (gatherTimeRoot>0.0) {
        printf("Allgatherv Lookahead:   MaxSize (%6d x %6d), Column %3d, Time (ms) = %8.3f, BW Est (GB/s) = %6.2f\n",
                nroot, nroot,
                MModAdd1(PANEL->pcol, PANEL->grid->npcol),
                gatherTimeRoot,
                nroot*nroot*8/(1.0E6*gatherTimeRoot));
      }
    }
  } else {

    int nq_max=0;
    MPI_Reduce(&n, &nq_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    // Total time for this col's scaller is the max over the col
    double scatter_time_total=0.;
    MPI_Reduce(&scatter_time[UPD], &scatter_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);

    // Min max avg over all cols' scatters
    double scatter_time_avg=0.;
    double scatter_time_min=0.;
    double scatter_time_max=0.;
    double scatter_time_stddev=0.;

    MPI_Allreduce(&scatter_time_total, &scatter_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->row_comm);
    MPI_Reduce(&scatter_time_total, &scatter_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->row_comm);
    MPI_Reduce(&scatter_time_total, &scatter_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    scatter_time_avg /= PANEL->grid->npcol;

    double scatter_var = (scatter_time_total - scatter_time_avg)*(scatter_time_total - scatter_time_avg);
    MPI_Reduce(&scatter_var, &scatter_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
    scatter_time_stddev /= PANEL->grid->npcol;
    scatter_time_stddev = sqrt(scatter_time_stddev);


    // Total time for this col's scaller is the max over the col
    double gather_time_total=0.;
    MPI_Reduce(&gather_time[UPD], &gather_time_total, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);

    // Min max avg over all cols' gathers
    double gather_time_avg=0.;
    double gather_time_min=0.;
    double gather_time_max=0.;
    double gather_time_stddev=0.;

    MPI_Allreduce(&gather_time_total, &gather_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->row_comm);
    MPI_Reduce(&gather_time_total, &gather_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gather_time_total, &gather_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    gather_time_avg /= PANEL->grid->npcol;

    double gather_var = (gather_time_total - gather_time_avg)*(gather_time_total - gather_time_avg);
    MPI_Reduce(&gather_var, &gather_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->row_comm);
    gather_time_stddev /= PANEL->grid->npcol;
    gather_time_stddev = sqrt(gather_time_stddev);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0) {
      if (scatter_time_max>0.0) {
        printf("Scatterv %s:             MaxSize (%6d x %6d), RootRow %2d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
                name.c_str(), nq_max, PANEL->jb, PANEL->prow, scatter_time_min, scatter_time_avg, scatter_time_stddev, scatter_time_max,
                nq_max*PANEL->jb*8/(1.0E6*scatter_time_avg));
      }

      if (gather_time_max>0.0) {
        printf("Allgatherv %s:           MaxSize (%6d x %6d), RootRow %2d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
                name.c_str(), nq_max, PANEL->jb, PANEL->prow, gather_time_min, gather_time_avg, gather_time_stddev, gather_time_max,
                nq_max*PANEL->jb*8/(1.0E6*gather_time_avg));
      }
    }
  }
}

void print_rowscatter_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {

  int n=0;
  std::string name;
  if(UPD == HPL_LOOK_AHEAD) {
    n   = PANEL->nu0;
    name = "Look Ahead";
  } else if(UPD == HPL_UPD_1) {
    n   = PANEL->nu1;
    name = "1";
  } else if(UPD == HPL_UPD_2) {
    n   = PANEL->nu2;
    name = "2";
  }

  double rowscatter_time_avg=0.;
  double rowscatter_time_min=0.;
  double rowscatter_time_max=0.;
  double rowscatter_time_stddev=0.;

  if (UPD == HPL_LOOK_AHEAD) {
    // Get time from GPU
    float rowscatterTime=0.;
    if (PANEL->grid->mycol==MModAdd1(PANEL->pcol, PANEL->grid->npcol)) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&rowscatterTime,
                                      rowScatterStart[UPD],
                                      rowScatterStop[UPD]));
    }

    double rowscatter_time = rowscatterTime;

    // min max avg over the column
    MPI_Allreduce(&rowscatter_time, &rowscatter_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
    MPI_Reduce(&rowscatter_time, &rowscatter_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
    MPI_Reduce(&rowscatter_time, &rowscatter_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
    rowscatter_time_avg /= PANEL->grid->nprow;

    double rowscatter_var = (rowscatter_time - rowscatter_time_avg)*(rowscatter_time - rowscatter_time_avg);
    MPI_Reduce(&rowscatter_var, &rowscatter_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
    rowscatter_time_stddev /= PANEL->grid->nprow;
    rowscatter_time_stddev = sqrt(rowscatter_time_stddev);


    double rowscatter_time_avg_root=0.;
    double rowscatter_time_min_root=0.;
    double rowscatter_time_max_root=0.;
    double rowscatter_time_stddev_root=0.;
    int n_root = 0;

    // Only one column should have done the lookahead, so just max over the row
    MPI_Reduce(&rowscatter_time_avg, &rowscatter_time_avg_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowscatter_time_min, &rowscatter_time_min_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowscatter_time_max, &rowscatter_time_max_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&rowscatter_time_stddev, &rowscatter_time_stddev_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&n, &n_root, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    double nbytes=0;
    if (PANEL->grid->nprow==1) {
      nbytes = 2*n_root*PANEL->jb*8.0;
    } else {
      nbytes = 2*n_root*PANEL->jb*8.0 + 2*n_root*PANEL->jb*8.0/PANEL->grid->nprow;
    }

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_root>0) {
      printf("RowScatter Lookahead:   MaxSize (%6d x %6d), Column %3d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              PANEL->jb, n_root, MModAdd1(PANEL->pcol, PANEL->grid->npcol), rowscatter_time_min_root, rowscatter_time_avg_root, rowscatter_time_stddev_root, rowscatter_time_max_root,
              nbytes/(1.0E6*rowscatter_time_avg_root));
    }
  } else {
    float rowscatterTime=0.;
    CHECK_HIP_ERROR(hipEventElapsedTime(&rowscatterTime,
                                        rowScatterStart[UPD],
                                        rowScatterStop[UPD]));

    double rowscatter_time = rowscatterTime;

    int n_max=0;
    MPI_Reduce(&n, &n_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);

    // Min max avg over all ranks
    MPI_Allreduce(&rowscatter_time, &rowscatter_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->all_comm);
    MPI_Reduce(&rowscatter_time, &rowscatter_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->all_comm);
    MPI_Reduce(&rowscatter_time, &rowscatter_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->all_comm);
    rowscatter_time_avg /= PANEL->grid->nprocs;

    double rowscatter_var = (rowscatter_time - rowscatter_time_avg)*(rowscatter_time - rowscatter_time_avg);
    MPI_Reduce(&rowscatter_var, &rowscatter_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->all_comm);
    rowscatter_time_stddev /= PANEL->grid->nprocs;
    rowscatter_time_stddev = sqrt(rowscatter_time_stddev);

    double nbytes=0;
    if (PANEL->grid->nprow==1) {
      nbytes = 2*n_max*PANEL->jb*8.0;
    } else {
      nbytes = 2*n_max*PANEL->jb*8.0 + 2*n_max*PANEL->jb*8.0/PANEL->grid->nprow;
    }

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_max>0) {
      printf("RowScatter %s:           MaxSize (%6d x %6d),             Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), BW Est (GB/s) = %6.2f\n",
              name.c_str(), PANEL->jb, n_max, rowscatter_time_min, rowscatter_time_avg, rowscatter_time_stddev, rowscatter_time_max,
              nbytes/(1.0E6*rowscatter_time_avg));
    }
  }
}

void print_update_stats(HPL_T_panel* PANEL, const HPL_T_UPD UPD) {

  int jb = PANEL->jb;

  int n=0;
  std::string name;
  if(UPD == HPL_LOOK_AHEAD) {
    n   = PANEL->nu0;
    name = "Look Ahead";
  } else if(UPD == HPL_UPD_1) {
    n   = PANEL->nu1;
    name = "1";
  } else if(UPD == HPL_UPD_2) {
    n   = PANEL->nu2;
    name = "2";
  }

  const int curr = (PANEL->grid->myrow == PANEL->prow ? 1 : 0);
  const int m   = PANEL->mp - (curr != 0 ? jb : 0);

  double trsm_time_avg=0.;
  double trsm_time_min=0.;
  double trsm_time_max=0.;
  double trsm_time_stddev=0.;

  double gemm_time_avg=0.;
  double gemm_time_min=0.;
  double gemm_time_max=0.;
  double gemm_time_stddev=0.;

  if (UPD==HPL_LOOK_AHEAD) {

    float trsmTime=0.;
    float gemmTime=0.;

    if (PANEL->grid->mycol==MModAdd1(PANEL->pcol, PANEL->grid->npcol)) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmTime,
                                        dtrsmStart[UPD],
                                        dtrsmStop[UPD]));

      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmTime,
                                          dgemmStart[UPD],
                                          dgemmStop[UPD]));
    }

    double trsm_time = trsmTime;
    double gemm_time = gemmTime;

    // min max avg over the column
    MPI_Allreduce(&trsm_time, &trsm_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
    MPI_Reduce(&trsm_time, &trsm_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
    MPI_Reduce(&trsm_time, &trsm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
    trsm_time_avg /= PANEL->grid->nprow;

    double trsm_var = (trsm_time - trsm_time_avg)*(trsm_time - trsm_time_avg);
    MPI_Reduce(&trsm_var, &trsm_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
    trsm_time_stddev /= PANEL->grid->nprow;
    trsm_time_stddev = sqrt(trsm_time_stddev);

    MPI_Allreduce(&gemm_time, &gemm_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->col_comm);
    MPI_Reduce(&gemm_time, &gemm_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->col_comm);
    MPI_Reduce(&gemm_time, &gemm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->col_comm);
    gemm_time_avg /= PANEL->grid->nprow;

    double gemm_var = (gemm_time - gemm_time_avg)*(gemm_time - gemm_time_avg);
    MPI_Reduce(&gemm_var, &gemm_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->col_comm);
    gemm_time_stddev /= PANEL->grid->nprow;
    gemm_time_stddev = sqrt(gemm_time_stddev);


    double trsm_time_avg_root=0.;
    double trsm_time_min_root=0.;
    double trsm_time_max_root=0.;
    double trsm_time_stddev_root=0.;
    double gemm_time_avg_root=0.;
    double gemm_time_min_root=0.;
    double gemm_time_max_root=0.;
    double gemm_time_stddev_root=0.;
    int n_root = 0;
    int m_root = 0;

    // Only one column should have done the lookahead, so just max over the row
    MPI_Reduce(&trsm_time_avg, &trsm_time_avg_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&trsm_time_min, &trsm_time_min_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&trsm_time_max, &trsm_time_max_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&trsm_time_stddev, &trsm_time_stddev_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gemm_time_avg, &gemm_time_avg_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gemm_time_min, &gemm_time_min_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gemm_time_max, &gemm_time_max_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&gemm_time_stddev, &gemm_time_stddev_root, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&n, &n_root, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);
    MPI_Reduce(&m, &m_root, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_root>0) {
      printf("Update Lookahead DTRSM: MaxSize (%6d x %6d), Column %3d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), TFLOPS Est = %6.2f\n",
              PANEL->jb, n_root, MModAdd1(PANEL->pcol, PANEL->grid->npcol), trsm_time_min_root, trsm_time_avg_root, trsm_time_stddev_root, trsm_time_max_root,
              (jb * jb * jb / 3.0 + 2.0 * n_root * jb * jb)/(1.0E9*trsm_time_avg_root));
      printf("Update Lookahead DGEMM: MaxSize (%6d x %6d), Column %3d, Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), TFLOPS Est = %6.2f\n",
              m_root, n_root, MModAdd1(PANEL->pcol, PANEL->grid->npcol), gemm_time_min_root, gemm_time_avg_root, gemm_time_stddev_root, gemm_time_max_root,
              (2.0 * m_root * n_root * jb)/(1.0E9*gemm_time_avg_root));
    }
  } else {

    float trsmTime=0.;
    float gemmTime=0.;

    if (n>0 && m>0) {
      CHECK_HIP_ERROR(hipEventElapsedTime(&trsmTime,
                                        dtrsmStart[UPD],
                                        dtrsmStop[UPD]));

      CHECK_HIP_ERROR(hipEventElapsedTime(&gemmTime,
                                          dgemmStart[UPD],
                                          dgemmStop[UPD]));
    }

    double trsm_time = trsmTime;
    double gemm_time = gemmTime;

    int n_max=0;
    MPI_Reduce(&n, &n_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->row_comm);
    int m_max=0;
    MPI_Reduce(&m, &m_max, 1, MPI_INT, MPI_MAX, 0, PANEL->grid->col_comm);

    // Min max avg over all ranks
    MPI_Allreduce(&trsm_time, &trsm_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->all_comm);
    MPI_Reduce(&trsm_time, &trsm_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->all_comm);
    MPI_Reduce(&trsm_time, &trsm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->all_comm);
    trsm_time_avg /= PANEL->grid->nprocs;

    double trsm_var = (trsm_time - trsm_time_avg)*(trsm_time - trsm_time_avg);
    MPI_Reduce(&trsm_var, &trsm_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->all_comm);
    trsm_time_stddev /= PANEL->grid->nprocs;
    trsm_time_stddev = sqrt(trsm_time_stddev);


    MPI_Allreduce(&gemm_time, &gemm_time_avg, 1, MPI_DOUBLE, MPI_SUM, PANEL->grid->all_comm);
    MPI_Reduce(&gemm_time, &gemm_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, PANEL->grid->all_comm);
    MPI_Reduce(&gemm_time, &gemm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, PANEL->grid->all_comm);
    gemm_time_avg /= PANEL->grid->nprocs;

    double gemm_var = (gemm_time - gemm_time_avg)*(gemm_time - gemm_time_avg);
    MPI_Reduce(&gemm_var, &gemm_time_stddev, 1, MPI_DOUBLE, MPI_SUM, 0, PANEL->grid->all_comm);
    gemm_time_stddev /= PANEL->grid->nprocs;
    gemm_time_stddev = sqrt(gemm_time_stddev);


    if (PANEL->grid->mycol==0 && PANEL->grid->myrow==0 && n_max>0 && m_max>0) {
      printf("Update %s DTRSM:         MaxSize (%6d x %6d),             Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), TFLOPS Est = %6.2f\n",
              name.c_str(), PANEL->jb, n_max, trsm_time_min, trsm_time_avg, trsm_time_stddev, trsm_time_max,
              (jb * jb * jb / 3.0 + 2.0 * n_max * jb * jb)/(1.0E9*trsm_time_avg));
      printf("Update %s DGEMM:         MaxSize (%6d x %6d),             Time (ms) (min, avg+/-stddev, max) = (%8.3f, %8.3f +/-%8.3f, %8.3f), TFLOPS Est = %6.2f\n",
              name.c_str(), m_max, n_max, gemm_time_min, gemm_time_avg, gemm_time_stddev, gemm_time_max,
              (2.0 * m_max * n_max * jb)/(1.0E9*gemm_time_avg));
    }
  }

}
