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

#include <limits>
#include "hpl.hpp"

void HPL_pdtest(HPL_T_test* TEST,
                HPL_T_grid* GRID,
                HPL_T_palg* ALGO,
                const int   N,
                const int   NB) {
/*
 * Purpose
 * =======
 *
 * HPL_pdtest performs  one  test  given a set of parameters such as the
 * process grid, the  problem size, the distribution blocking factor ...
 * This function generates  the data, calls  and times the linear system
 * solver,  checks  the  accuracy  of the  obtained vector solution  and
 * writes this information to the file pointed to by TEST->outfp.
 *
 * Arguments
 * =========
 *
 * TEST    (global input)                HPL_T_test *
 *         On entry,  TEST  points  to a testing data structure:  outfp
 *         specifies the output file where the results will be printed.
 *         It is only defined and used by the process  0  of the  grid.
 *         thrsh  specifies  the  threshhold value  for the test ratio.
 *         Concretely, a test is declared "PASSED"  if and only if the
 *         following inequality is satisfied:
 *         ||Ax-b||_oo / ( epsil *
 *                         ( || x ||_oo * || A ||_oo + || b ||_oo ) *
 *                          N )  < thrsh.
 *         epsil  is the  relative machine precision of the distributed
 *         computer. Finally the test counters, kfail, kpass, kskip and
 *         ktest are updated as follows:  if the test passes,  kpass is
 *         incremented by one;  if the test fails, kfail is incremented
 *         by one; if the test is skipped, kskip is incremented by one.
 *         ktest is left unchanged.
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters to be used for this test.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the order of the coefficient matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 *
 * ---------------------------------------------------------------------
 */
/*
 * .. Local Variables ..
 */
#ifdef HPL_DETAILED_TIMING
  double HPL_w[HPL_TIMING_N];
#endif
  HPL_T_pmat mat;
  double     wtime[1];
  int        ierr;
  double     Anorm1, AnormI, Gflops, Xnorm1, XnormI, BnormI, resid0, resid1;
  double*    Bptr;
  double*    dBptr;
  static int first = 1;
  int        ii, ip2, mycol, myrow, npcol, nprow, nq;
  char       ctop, cpfact, crfact;
  time_t     current_time_start, current_time_end;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  /*
   * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
   * N by N+1.  One column is added in every process column for the solve.
   * The  result  however  is stored in a 1 x N vector replicated in every
   * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
   * workspace is mp.
   */
  ierr = HPL_pdmatgen(TEST, GRID, ALGO, &mat, N, NB);

  if(ierr != HPL_SUCCESS) {
    (TEST->kskip)++;
    HPL_pdmatfree(&mat);
    return;
  }

  /* Create row-swapping data type */
  MPI_Type_contiguous(NB + 4, MPI_DOUBLE, &PDFACT_ROW);
  MPI_Type_commit(&PDFACT_ROW);

  /*
   * generate matrix and right-hand-side, [ A | b ] which is N by N+1.
   */
  HPL_pdrandmat(GRID, N, N + 1, NB, mat.dA, mat.ld, HPL_ISEED);

  /*
   * Solve linear system
   */
  HPL_ptimer_boot();
  (void)HPL_barrier(GRID->all_comm);
  time(&current_time_start);
  HPL_ptimer(0);
  HPL_pdgesv(GRID, ALGO, &mat);
  HPL_ptimer(0);
  time(&current_time_end);

  /*
   * Gather max of all CPU and WALL clock timings and print timing results
   */
  HPL_ptimer_combine(
      GRID->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME, 1, 0, wtime);

  if((myrow == 0) && (mycol == 0)) {
    if(first) {
      HPL_fprintf(TEST->outfp,
                  "%s%s\n",
                  "========================================",
                  "========================================");
      HPL_fprintf(TEST->outfp,
                  "%s%s\n",
                  "T/V                N    NB     P     Q",
                  "               Time                 Gflops");
      HPL_fprintf(TEST->outfp,
                  "%s%s\n",
                  "----------------------------------------",
                  "----------------------------------------");
      if(TEST->thrsh <= HPL_rzero) first = 0;
    }
    /*
     * 2/3 N^3 - 1/2 N^2 flops for LU factorization + 2 N^2 flops for solve.
     * Print WALL time
     */
    Gflops = (((double)(N) / 1.0e+9) * ((double)(N) / wtime[0])) *
             ((2.0 / 3.0) * (double)(N) + (3.0 / 2.0));

    cpfact = (((HPL_T_FACT)(ALGO->pfact) == (HPL_T_FACT)(HPL_LEFT_LOOKING))
                  ? (char)('L')
                  : (((HPL_T_FACT)(ALGO->pfact) == (HPL_T_FACT)(HPL_CROUT))
                         ? (char)('C')
                         : (char)('R')));
    crfact = (((HPL_T_FACT)(ALGO->rfact) == (HPL_T_FACT)(HPL_LEFT_LOOKING))
                  ? (char)('L')
                  : (((HPL_T_FACT)(ALGO->rfact) == (HPL_T_FACT)(HPL_CROUT))
                         ? (char)('C')
                         : (char)('R')));

    if(ALGO->btopo == HPL_1RING)
      ctop = '0';
    else if(ALGO->btopo == HPL_1RING_M)
      ctop = '1';
    else if(ALGO->btopo == HPL_2RING)
      ctop = '2';
    else if(ALGO->btopo == HPL_2RING_M)
      ctop = '3';
    else if(ALGO->btopo == HPL_BLONG)
      ctop = '4';
    else /* if( ALGO->btopo == HPL_BLONG_M ) */
      ctop = '5';

    if(wtime[0] > HPL_rzero) {
      HPL_fprintf(TEST->outfp,
                  "W%c%1d%c%c%1d%c%1d%12d %5d %5d %5d %18.2f     %18.3e\n",
                  (GRID->order == HPL_ROW_MAJOR ? 'R' : 'C'),
                  ALGO->depth,
                  ctop,
                  crfact,
                  ALGO->nbdiv,
                  cpfact,
                  ALGO->nbmin,
                  N,
                  NB,
                  nprow,
                  npcol,
                  wtime[0],
                  Gflops);
      HPL_fprintf(TEST->outfp,
                  "HPL_pdgesv() start time %s\n",
                  ctime(&current_time_start));
      HPL_fprintf(TEST->outfp,
                  "HPL_pdgesv() end time   %s\n",
                  ctime(&current_time_end));
    }
#ifdef HPL_PROGRESS_REPORT
    printf("Final Score:    %7.4e GFLOPS \n", Gflops);
#endif
  }
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer_combine(GRID->all_comm,
                     HPL_AMAX_PTIME,
                     HPL_WALL_PTIME,
                     HPL_TIMING_N,
                     HPL_TIMING_BEG,
                     HPL_w);
  if((myrow == 0) && (mycol == 0)) {
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-");
    /*
     * Lbcast
     */
    if(HPL_w[HPL_TIMING_LBCAST - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "Max aggregated wall time bcast . . . : %18.2f\n",
                  HPL_w[HPL_TIMING_LBCAST - HPL_TIMING_BEG]);
    /*
     * Panel copy
     */
    if(HPL_w[HPL_TIMING_COPY - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "+ Max aggregated wall time panel copy: %18.2f\n",
                  HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG]);
    /*
     * Recursive panel factorization
     */
    if(HPL_w[HPL_TIMING_RPFACT - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "+ Max aggregated wall time rfact . . : %18.2f\n",
                  HPL_w[HPL_TIMING_RPFACT - HPL_TIMING_BEG]);
    /*
     * Panel factorization
     */
    if(HPL_w[HPL_TIMING_PFACT - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "+ + Max aggregated wall time pfact . : %18.2f\n",
                  HPL_w[HPL_TIMING_PFACT - HPL_TIMING_BEG]);
    /*
     * Panel factorization (swap)
     */
    if(HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "+ + Max aggregated wall time mxswp . : %18.2f\n",
                  HPL_w[HPL_TIMING_MXSWP - HPL_TIMING_BEG]);
    /*
     * Update (swap)
     */
    if(HPL_w[HPL_TIMING_LASWP - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "Max aggregated wall time laswp . . . : %18.2f\n",
                  HPL_w[HPL_TIMING_LASWP - HPL_TIMING_BEG]);
    /*
     * Update
     */
    if(HPL_w[HPL_TIMING_UPDATE - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "Max aggregated wall time update  . . : %18.2f\n",
                  HPL_w[HPL_TIMING_UPDATE - HPL_TIMING_BEG]);
    /*
     * Upper triangular system solve
     */
    if(HPL_w[HPL_TIMING_PTRSV - HPL_TIMING_BEG] > HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "Max aggregated wall time up tr sv  . : %18.2f\n",
                  HPL_w[HPL_TIMING_PTRSV - HPL_TIMING_BEG]);

    if(TEST->thrsh <= HPL_rzero)
      HPL_fprintf(TEST->outfp,
                  "%s%s\n",
                  "========================================",
                  "========================================");
  }
#endif

  /* Release row swapping datatype */
  MPI_Type_free(&PDFACT_ROW);

  /*
   * Quick return, if I am not interested in checking the computations
   */
  if(TEST->thrsh <= HPL_rzero) {
    (TEST->kpass)++;
    HPL_pdmatfree(&mat);
    return;
  }
  /*
   * Check info returned by solve
   */
  if(mat.info != 0) {
    if((myrow == 0) && (mycol == 0))
      HPL_pwarn(TEST->outfp,
                __LINE__,
                "HPL_pdtest",
                "%s %d, %s",
                "Error code returned by solve is",
                mat.info,
                "skip");
    (TEST->kskip)++;
    HPL_pdmatfree(&mat);
    return;
  }
  /*
   * Check computation, re-generate [ A | b ], compute norm 1 and inf of A and
   * x, and norm inf of b - A x. Display residual checks.
   */
  HPL_pdrandmat(GRID, N, N + 1, NB, mat.dA, mat.ld, HPL_ISEED);

  Anorm1 = HPL_pdlange(GRID, HPL_NORM_1, N, N, NB, mat.dA, mat.ld);
  AnormI = HPL_pdlange(GRID, HPL_NORM_I, N, N, NB, mat.dA, mat.ld);
  /*
   * Because x is distributed in process rows, switch the norms
   */
  XnormI = HPL_pdlange(GRID, HPL_NORM_1, 1, N, NB, mat.dX, 1);
  Xnorm1 = HPL_pdlange(GRID, HPL_NORM_I, 1, N, NB, mat.dX, 1);
  /*
   * If I am in the col that owns b, (1) compute local BnormI, (2) all_reduce to
   * find the max (in the col). Then (3) broadcast along the rows so that every
   * process has BnormI. Note that since we use a uniform distribution in
   * [-0.5,0.5] for the entries of B, it is very likely that BnormI (<=,~) 0.5.
   */

  // Bptr  = Mptr( mat.A , 0, nq, mat.ld );
  size_t BptrBytes = Mmax(mat.nq, mat.ld) * sizeof(double);
  Bptr             = (double*)malloc(BptrBytes);

  nq    = HPL_numroc(N, NB, NB, mycol, 0, npcol);
  dBptr = Mptr(mat.dA, 0, nq, mat.ld);
  if(mycol == HPL_indxg2p(N, NB, NB, 0, npcol)) {
    if(mat.mp > 0) {
      // int id = HPL_idamax( mat.mp, Bptr, 1);
      // BnormI = Bptr[id];
      int id;
      CHECK_ROCBLAS_ERROR(rocblas_idamax(handle, mat.mp, dBptr, 1, &id));

      // Note: id is in Fortran indexing
      CHECK_HIP_ERROR(hipMemcpy(
          &BnormI, dBptr + id - 1, 1 * sizeof(double), hipMemcpyDeviceToHost));
      BnormI = Mabs(BnormI);
    } else {
      BnormI = HPL_rzero;
    }
    (void)HPL_all_reduce(
        (void*)(&BnormI), 1, HPL_DOUBLE, HPL_MAX, GRID->col_comm);
  }
  (void)HPL_broadcast((void*)(&BnormI),
                      1,
                      HPL_DOUBLE,
                      HPL_indxg2p(N, NB, NB, 0, npcol),
                      GRID->row_comm);
  /*
   * If I own b, compute ( b - A x ) and ( - A x ) otherwise
   */

  // rocBLAS < v4.2 has an integer overflow problem in dgemv, so
  // chunk the nq columns to compute the full dgemv
  const int nq_chunk = std::numeric_limits<int>::max() / (mat.ld);

  if(mycol == HPL_indxg2p(N, NB, NB, 0, npcol)) {
    const double one  = 1.0;
    const double mone = -1.0;

    for(int nn = 0; nn < nq; nn += nq_chunk) {
      int nb = Mmin(nq - nn, nq_chunk);
      CHECK_ROCBLAS_ERROR(rocblas_dgemv(handle,
                                        rocblas_operation_none,
                                        mat.mp,
                                        nb,
                                        &mone,
                                        Mptr(mat.dA, 0, nn, mat.ld),
                                        mat.ld,
                                        Mptr(mat.dX, 0, nn, 1),
                                        1,
                                        &one,
                                        dBptr,
                                        1));
    }

    CHECK_HIP_ERROR(
        hipMemcpy(Bptr, dBptr, mat.mp * sizeof(double), hipMemcpyDeviceToHost));
  } else if(nq > 0) {
    const double one  = 1.0;
    const double zero = 0.0;
    const double mone = -1.0;

    int nb = Mmin(nq, nq_chunk);
    CHECK_ROCBLAS_ERROR(rocblas_dgemv(handle,
                                      rocblas_operation_none,
                                      mat.mp,
                                      nb,
                                      &mone,
                                      Mptr(mat.dA, 0, 0, mat.ld),
                                      mat.ld,
                                      Mptr(mat.dX, 0, 0, 1),
                                      1,
                                      &zero,
                                      dBptr,
                                      1));

    for(int nn = nb; nn < nq; nn += nq_chunk) {
      int nb = Mmin(nq - nn, nq_chunk);
      CHECK_ROCBLAS_ERROR(rocblas_dgemv(handle,
                                        rocblas_operation_none,
                                        mat.mp,
                                        nb,
                                        &mone,
                                        Mptr(mat.dA, 0, nn, mat.ld),
                                        mat.ld,
                                        Mptr(mat.dX, 0, nn, 1),
                                        1,
                                        &one,
                                        dBptr,
                                        1));
    }

    CHECK_HIP_ERROR(
        hipMemcpy(Bptr, dBptr, mat.mp * sizeof(double), hipMemcpyDeviceToHost));
  } else {
    for(ii = 0; ii < mat.mp; ii++) Bptr[ii] = HPL_rzero;
  }
  /*
   * Reduce the distributed residual in process column 0
   */
  if(mat.mp > 0)
    (void)HPL_reduce(Bptr, mat.mp, HPL_DOUBLE, HPL_SUM, 0, GRID->row_comm);

  /*
   * Compute || b - A x ||_oo
   */
  CHECK_HIP_ERROR(
      hipMemcpy(dBptr, Bptr, mat.mp * sizeof(double), hipMemcpyHostToDevice));
  resid0 = HPL_pdlange(GRID, HPL_NORM_I, N, 1, NB, dBptr, mat.ld);
  /*
   * Computes and displays norms, residuals ...
   */
  if(N <= 0) {
    resid1 = HPL_rzero;
  } else {
    resid1 = resid0 / (TEST->epsil * (AnormI * XnormI + BnormI) * (double)(N));
  }

  if(resid1 < TEST->thrsh)
    (TEST->kpass)++;
  else
    (TEST->kfail)++;

  if((myrow == 0) && (mycol == 0)) {
    HPL_fprintf(TEST->outfp,
                "%s%s\n",
                "----------------------------------------",
                "----------------------------------------");
    HPL_fprintf(TEST->outfp,
                "%s%16.7f%s%s\n",
                "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ",
                resid1,
                " ...... ",
                (resid1 < TEST->thrsh ? "PASSED" : "FAILED"));

    if(resid1 >= TEST->thrsh) {
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||Ax-b||_oo  . . . . . . . . . . . . . . . . . = ",
                  resid0);
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||A||_oo . . . . . . . . . . . . . . . . . . . = ",
                  AnormI);
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||A||_1  . . . . . . . . . . . . . . . . . . . = ",
                  Anorm1);
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||x||_oo . . . . . . . . . . . . . . . . . . . = ",
                  XnormI);
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||x||_1  . . . . . . . . . . . . . . . . . . . = ",
                  Xnorm1);
      HPL_fprintf(TEST->outfp,
                  "%s%18.6f\n",
                  "||b||_oo . . . . . . . . . . . . . . . . . . . = ",
                  BnormI);
    }

#ifdef HPL_PROGRESS_REPORT
    if(resid1 < TEST->thrsh)
      printf("Residual Check: PASSED \n");
    else
      printf("Residual Check: FAILED \n");
#endif
  }

  if(Bptr) free(Bptr);
  HPL_pdmatfree(&mat);
}
