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
#include <hip/hip_runtime.h>
#include "hpl_hip_ex.hpp"

using namespace hip_ex;

template <int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE) __global__
    void pdpanrlT(const int M,
                  const int NB,
                  const int JB,
                  double* __restrict__ A,
                  const int LDA,
                  const int curr,
                  const int myrow,
                  const int JJ,
                  double* __restrict__ L1,
                  int* __restrict__ loc_workspace,
                  double* __restrict__ max_workspace,
                  double* __restrict__ dev_workspace,
                  double* __restrict__ host_workspace,
                  atomic_ref<int32_t, thread_scope_system> host_flag,
                  barrier<thread_scope_device>             barrier) {

  const int t       = threadIdx.x;
  const int block   = blockIdx.x;
  const int Nblocks = gridDim.x - 1;

  __shared__ double s_max[BLOCKSIZE];
  __shared__ int    s_loc[BLOCKSIZE];
  __shared__ double s_work[BLOCKSIZE];

  int    loc;
  double max;

  // Device workspaces to hold pivoted rows at agent scope
  atomic_ref<double, thread_scope_device> d_max(dev_workspace[0]);
  atomic_ref<double, thread_scope_device> d_loc(dev_workspace[1]);
  atomic_ref<double, thread_scope_device> d_row(dev_workspace[3]);
  atomic_ref<double, thread_scope_device> d_acur(dev_workspace[4 + t]);
  double                                  amax, acur;

  // Workspace for agent level pivot candidates
  double* candidate_rows = dev_workspace + 4 + NB;
  atomic_ref<double, thread_scope_device> d_ccur(candidate_rows[t]);

  using arrival_token = hip_ex::barrier<thread_scope_device>::arrival_token;
  arrival_token bar;

  hip_ex::barrier<thread_scope_block> block_barrier;

  if(block == 0) {
    atomic_ref<double, thread_scope_system> h_max(host_workspace[0]);
    atomic_ref<double, thread_scope_system> h_loc(host_workspace[1]);
    atomic_ref<double, thread_scope_system> h_row(host_workspace[3]);
    atomic_ref<double, thread_scope_system> h_amax(host_workspace[4 + t]);
    atomic_ref<double, thread_scope_system> h_acur(host_workspace[4 + t + NB]);

    // loop through columns
    for(int jj = 0; jj < JB; ++jj) {
      // Wait for all blocks to have completed partial reduction
      if(t == 0)
        while(!barrier.is_last_to_arrive()) {}
      block_barrier.arrive_and_wait(memory_order_relaxed);

      // Complete maxloc reduction
      atomic_block_maxloc<BLOCKSIZE>(
          Nblocks, max_workspace + 1, s_max, s_loc, max, loc);
      loc += 1;

      if(t < NB) {
        // Read local candidate pivot row out of agent work buffer
        atomic_ref<double, thread_scope_device> d_cmax(
            candidate_rows[t + loc * NB]);
        amax = d_cmax.load(memory_order_relaxed);
        if(curr) acur = d_ccur.load(memory_order_relaxed);

        // write row into host swap space
        h_amax.store(amax, memory_order_relaxed);
        if(curr) h_acur.store(acur, memory_order_relaxed);
      }

      if(t == 0) {
        // Write row number and max val to header
        atomic_ref<int, thread_scope_device> d_wloc(loc_workspace[loc]);
        loc = d_wloc.load(memory_order_relaxed);

        h_max.store(static_cast<double>(max), memory_order_relaxed);
        h_loc.store(static_cast<double>(loc), memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      if(t == 0) {
        // Mark pivot row as found
        host_flag.store(1, memory_order_relaxed);

        // Wait for Host to unlock
        host_flag.wait(1, memory_order_relaxed);
      }
      block_barrier.arrive_and_wait(memory_order_relaxed);

      if(t < NB) {
        // Read pivot row out of host swap space
        amax = h_amax.load(memory_order_relaxed);
        acur = h_acur.load(memory_order_relaxed);

        // write row into agent work buffer
        atomic_ref<double, thread_scope_device> L1tj(L1[t + jj * NB]);
        L1tj.store(amax, memory_order_relaxed);
        d_acur.store(acur, memory_order_relaxed);
      }

      if(t == 0) {
        const double gmax   = h_max.load(memory_order_relaxed);
        const double srcloc = h_loc.load(memory_order_relaxed);
        const double srcrow = h_row.load(memory_order_relaxed);
        d_max.store(gmax, memory_order_relaxed);
        d_loc.store(srcloc, memory_order_relaxed);
        d_row.store(srcrow, memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      // Unblock the other blocks
      if(t == 0) bar = barrier.arrive(memory_order_relaxed);
    }

  } else {

    const int  m     = t + BLOCKSIZE * (block - 1);
    const bool bcurr = (block == 1) && curr;

    atomic_ref<double, thread_scope_device> d_cmax(
        candidate_rows[t + block * NB]);

    atomic_ref<int, thread_scope_device>    d_wloc(loc_workspace[block]);
    atomic_ref<double, thread_scope_device> d_wmax(max_workspace[block]);

    // pointer to current column
    double* An = A + JJ * LDA;

    // Each block does partial reduction to find pivot row
    int    loc = -1;
    double max = 0.;
    for(int mm = m; mm < M; mm += BLOCKSIZE * Nblocks) {
      const double Amn = An[mm];
      if(std::abs(Amn) > std::abs(max)) {
        loc = mm;
        max = Amn;
      }
    }
    block_maxloc<BLOCKSIZE>(max, loc, s_max, s_loc);

    if(t < NB) {
      // Write out to workspace
      d_cmax.store(A[loc + t * LDA], memory_order_relaxed);

      // write top row to workspace
      if(bcurr) d_ccur.store(A[0 + t * LDA], memory_order_relaxed);
    }

    if(t == 0) {
      d_wloc.store(loc, memory_order_relaxed);
      d_wmax.store(max, memory_order_relaxed);
    }

    waitcnt(thread_scope_device);
    block_barrier.arrive_and_wait(memory_order_relaxed);

    // Signal the first block that the workspace is populated and wait for the
    // swap
    if(t == 0) barrier.arrive_and_wait(memory_order_relaxed);
    block_barrier.arrive_and_wait(memory_order_relaxed);

    int ii = 0;

    for(int jj = 1; jj < JB; ++jj) {
      // Perform the row swap
      const double gmax   = d_max.load(memory_order_relaxed);
      const int    srcloc = d_loc.load(memory_order_relaxed);
      const int    srcrow = d_row.load(memory_order_relaxed);

      // shift down a row
      if(bcurr) ii++;

      double acmax;
      if(t < NB) {
        // Read in the Amax row to LDS
        atomic_ref<double, thread_scope_device> L1tj(L1[t + (jj - 1) * NB]);
        acmax     = L1tj.load(memory_order_relaxed);
        s_work[t] = acmax;

        if(myrow == srcrow) { // if this rank owns the src row
          for(int mm = BLOCKSIZE * (block - 1); mm < M;
              mm += BLOCKSIZE * Nblocks) {
            if(mm <= srcloc &&
               srcloc < mm + BLOCKSIZE) { // and my block owns the src row
              A[srcloc + t * LDA] =
                  d_acur.load(memory_order_relaxed); // perform the swap
            }
          }
        }
      }
      block_barrier.arrive_and_wait(memory_order_acq_rel);

      // Scale column by max value, update next column, and record pivot
      // candidates
      int    loc = -1;
      double max = 0.;

      for(int mm = m; mm < M; mm += BLOCKSIZE * Nblocks) {
        if(ii <= mm) {
          const double Amn   = An[mm] / gmax;
          const double Amnp1 = An[mm + LDA] - s_work[JJ + jj] * Amn;

          An[mm]       = Amn;
          An[mm + LDA] = Amnp1;

          if(std::abs(Amnp1) > std::abs(max)) {
            loc = mm;
            max = Amnp1;
          }
        }
      }

      // Each block does partial reduction to find pivot row
      block_maxloc<BLOCKSIZE>(max, loc, s_max, s_loc);

      if(t < NB) {
        amax = A[loc + t * LDA];

        // rank-1 update of swapped rows
        if(jj + JJ < t && t < JB + JJ) { amax -= acmax * An[loc]; }

        if(bcurr) {
          acur = A[ii + t * LDA];

          // rank-1 update of swapped rows
          if(jj + JJ < t && t < JB + JJ) { acur -= acmax * An[ii]; }
        }

        // Write out to workspace
        d_cmax.store(amax, memory_order_relaxed);

        // Write top row
        if(bcurr) d_ccur.store(acur, memory_order_relaxed);
      }

      if(t == 0) {
        d_wloc.store(loc, memory_order_relaxed);
        d_wmax.store(max, memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      // Signal the first block that the workspace is populated
      if(t == 0) bar = barrier.arrive(memory_order_relaxed);

      // Rank 1 update while the row is exchanged on the host
      for(int mm = m; mm < M; mm += BLOCKSIZE * Nblocks) {
        if(ii <= mm) {
          const double Amn = An[mm];
          for(int k = jj + 1; k < JB; ++k) {
            A[mm + (k + JJ) * LDA] -= s_work[JJ + k] * Amn;
          }
        }
      }

      // shift A one column
      An += LDA;

      if(t == 0) barrier.wait(std::move(bar), memory_order_relaxed);
      block_barrier.arrive_and_wait(memory_order_relaxed);
    }

    // Perform final row swap and update
    const double gmax   = d_max.load(memory_order_relaxed);
    const int    srcloc = d_loc.load(memory_order_relaxed);
    const int    srcrow = d_row.load(memory_order_relaxed);

    // shift down a row
    if(bcurr) ii++;

    if(myrow == srcrow && t < NB) { // if this rank owns the src row
      for(int mm = BLOCKSIZE * (block - 1); mm < M; mm += BLOCKSIZE * Nblocks) {
        if(mm <= srcloc &&
           srcloc < mm + BLOCKSIZE) { // and my block owns the src row
          A[srcloc + t * LDA] =
              d_acur.load(memory_order_relaxed); // perform the swap
        }
      }
    }
    block_barrier.arrive_and_wait(memory_order_relaxed);

    // Scale final column by max value
    for(int mm = m; mm < M; mm += BLOCKSIZE * Nblocks) {
      if(ii <= mm) { An[mm] /= gmax; }
    }
  }
}

void HPL_pdpanrlT(HPL_T_panel* PANEL,
                  const int    M,
                  const int    N,
                  const int    ICOFF) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdpanrlT factorizes  a panel of columns  that is a sub-array of a
   * larger one-dimensional panel A using the Right-looking variant of the
   * usual one-dimensional algorithm.  The lower triangular N0-by-N0 upper
   * block of the panel is stored in transpose form.
   *
   * Bi-directional  exchange  is  used  to  perform  the  swap::broadcast
   * operations  at once  for one column in the panel.  This  results in a
   * lower number of slightly larger  messages than usual.  On P processes
   * and assuming bi-directional links,  the running time of this function
   * can be approximated by (when N is equal to N0):
   *
   *    N0 * log_2( P ) * ( lat + ( 2*N0 + 4 ) / bdwth ) +
   *    N0^2 * ( M - N0/3 ) * gam2-3
   *
   * where M is the local number of rows of  the panel, lat and bdwth  are
   * the latency and bandwidth of the network for  double  precision  real
   * words,  and  gam2-3  is an estimate of the  Level 2 and Level 3  BLAS
   * rate of execution. The  recursive  algorithm  allows indeed to almost
   * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
   * large  number of modern machines,  this  operation is however latency
   * bound,  meaning  that its cost can  be estimated  by only the latency
   * portion N0 * log_2(P) * lat.  Mono-directional links will double this
   * communication cost.
   *
   * Note that  one  iteration of the the main loop is unrolled. The local
   * computation of the absolute value max of the next column is performed
   * just after its update by the current column. This allows to bring the
   * current column only  once through  cache at each  step.  The  current
   * implementation  does not perform  any blocking  for  this sequence of
   * BLAS operations, however the design allows for plugging in an optimal
   * (machine-specific) specialized  BLAS-like kernel.  This idea has been
   * suggested to us by Fred Gustavson, IBM T.J. Watson Research Center.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * M       (local input)                 const int
   *         On entry,  M specifies the local number of rows of sub(A).
   *
   * N       (local input)                 const int
   *         On entry,  N specifies the local number of columns of sub(A).
   *
   * ICOFF   (global input)                const int
   *         On entry, ICOFF specifies the row and column offset of sub(A)
   *         in A.
   *
   * ---------------------------------------------------------------------
   */

  double *A, *L1;
  int     curr, ii, jj, lda;

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PFACT);
#endif

  HPL_T_pmat* mat = PANEL->pmat;

  A    = PANEL->A0;
  lda  = PANEL->lda0;
  L1   = PANEL->L1;
  curr = PANEL->grid->myrow == PANEL->prow ? 1 : 0;

  HPL_T_grid* grid  = PANEL->grid;
  MPI_Comm    comm  = grid->col_comm;
  int         myrow = grid->myrow;
  int         nprow = grid->nprow;

  jj = ICOFF;
  if(curr != 0) {
    ii = ICOFF;
  } else {
    ii = 0;
  }

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  constexpr int BLOCKSIZE = 1024;

  int Nblocks =
      std::min(mat->pfact_max_blocks, (M + BLOCKSIZE - 1) / BLOCKSIZE + 1);

  barrier<thread_scope_device> barrier(mat->barrier_space, Nblocks);

  atomic_ref<int32_t, thread_scope_system> host_flag(mat->host_flag[0]);
  host_flag.store(0, memory_order_relaxed);

  if(M > 0) {
    int     m     = M;
    int     n     = N;
    double* Aptr  = A + ii;
    double* L1ptr = L1 + jj * PANEL->jb;

    void* params[] = {&m,
                      &(PANEL->jb),
                      &n,
                      &Aptr,
                      &lda,
                      &curr,
                      &myrow,
                      &jj,
                      &L1ptr,
                      &(mat->loc_workspace),
                      &(mat->max_workspace),
                      &(mat->dev_workspace),
                      &(mat->host_workspace),
                      &host_flag,
                      &barrier};
    CHECK_HIP_ERROR(hipLaunchCooperativeKernel(pdpanrlT<BLOCKSIZE>,
                                               dim3(Nblocks),
                                               dim3(BLOCKSIZE),
                                               params,
                                               0,
                                               stream));
  }

  int     cnt0  = 4 + 2 * PANEL->jb;
  double* WORK  = mat->host_workspace;
  double* Wwork = WORK + cnt0;

  for(int j = 0; j < N; j++) {
    /*Wait for host_flag to update from GPU*/
    if(M > 0) host_flag.wait(0, memory_order_acquire);

    HPL_pdmxswp(PANEL, M, ii, jj + j, WORK);

    /*Signal GPU*/
    if(M > 0) host_flag.store(0, memory_order_release);
  }

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PFACT);
#endif
}
