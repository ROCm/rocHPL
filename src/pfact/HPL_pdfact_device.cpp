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
#include <hip/hip_runtime.h>
#include "hpl_hip_ex.hpp"

using namespace hip_ex;

template<int BLOCKSIZE>
__launch_bounds__(BLOCKSIZE,8)
__global__ void pdfact(const int M,
                       const int NB,
                       const int JB,
                       double   *__restrict__ A,
                       const int LDA,
                       const int curr,
                       const int myrow,
                       const int JJ,
                       double   *__restrict__ L1,
                       int      *__restrict__ loc_workspace,
                       double   *__restrict__ max_workspace,
                       double   *__restrict__ dev_workspace,
                       double   *__restrict__ host_workspace,
                       atomic_ref<int32_t, thread_scope_system> host_flag,
                       barrier<thread_scope_device> barrier) {

  const int t = threadIdx.x;
  const int block = blockIdx.x;

  __shared__ double s_Amax[BLOCKSIZE];
  __shared__ double s_max[BLOCKSIZE];
  __shared__ int    s_loc[BLOCKSIZE];

  int    loc;
  double max;

  //Device workspaces to hold pivoted rows at agent scope
  atomic_ref<double, thread_scope_device> d_max(dev_workspace[0]);
  atomic_ref<double, thread_scope_device> d_loc(dev_workspace[1]);
  atomic_ref<double, thread_scope_device> d_row(dev_workspace[3]);
  atomic_ref<double, thread_scope_device> d_amax(dev_workspace[4 + t]);
  atomic_ref<double, thread_scope_device> d_acur(dev_workspace[4 + t + NB]);
  double amax, acur;

  //Workspace for agent level pivot candidates
  double *candidate_rows = dev_workspace + 4 + 2 * NB;
  atomic_ref<double, thread_scope_device> d_ccur(candidate_rows[t]);

  using arrival_token = hip_ex::barrier<thread_scope_device>::arrival_token;
  arrival_token bar;

  hip_ex::barrier<thread_scope_block> block_barrier;

  if (block==0) {
    atomic_ref<double, thread_scope_system> h_max(host_workspace[0]);
    atomic_ref<double, thread_scope_system> h_loc(host_workspace[1]);
    atomic_ref<double, thread_scope_system> h_row(host_workspace[3]);
    atomic_ref<double, thread_scope_system> h_amax(host_workspace[4+t]);
    atomic_ref<double, thread_scope_system> h_acur(host_workspace[4+t+NB]);

    //loop through columns
    for (int jj = 0; jj < JB; ++jj) {
      //Wait for all blocks to have completed partial reduction
      if (t==0) while (!barrier.is_last_to_arrive()) {}
      block_barrier.arrive_and_wait(memory_order_relaxed);

      // Complete maxloc reduction
      atomic_block_maxloc<BLOCKSIZE>(gridDim.x - 1, max_workspace + 1, s_max, s_loc, max, loc);
      loc+=1;

      if (t<NB) {
        //Read local candidate pivot row out of agent work buffer
        atomic_ref<double, thread_scope_device> d_cmax(candidate_rows[t + loc * NB]);
        amax = d_cmax.load(memory_order_relaxed);
        if (curr) acur = d_ccur.load(memory_order_relaxed);

        //write row into host swap space
        h_amax.store(amax, memory_order_relaxed);
        if (curr) h_acur.store(acur, memory_order_relaxed);
      }

      if(t==0) {
        //Write row number and max val to header
        atomic_ref<int, thread_scope_device> d_wloc(loc_workspace[loc]);
        loc = d_wloc.load(memory_order_relaxed);

        h_max.store(static_cast<double>(max), memory_order_relaxed);
        h_loc.store(static_cast<double>(loc), memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      if(t==0) {
        //Mark pivot row as found
        host_flag.store(1, memory_order_relaxed);

        //Wait for Host to unlock
        host_flag.wait(1, memory_order_relaxed);
      }
      block_barrier.arrive_and_wait(memory_order_relaxed);

      if (t<NB) {
        //Read pivot row out of host swap space
        amax = h_amax.load(memory_order_relaxed);
        acur = h_acur.load(memory_order_relaxed);

        //write row into agent work buffer
        d_amax.store(amax, memory_order_relaxed);
        d_acur.store(acur, memory_order_relaxed);
      }

      if (t==0) {
        const double gmax   = h_max.load(memory_order_relaxed);
        const double srcloc = h_loc.load(memory_order_relaxed);
        const double srcrow = h_row.load(memory_order_relaxed);
        d_max.store(gmax,   memory_order_relaxed);
        d_loc.store(srcloc, memory_order_relaxed);
        d_row.store(srcrow, memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      //Unblock the other blocks
      if (t==0) bar = barrier.arrive(memory_order_relaxed);

      if (t<NB) {
        L1[t + jj*NB] = amax;
      }
    }

  } else {

    const int m = t + BLOCKSIZE * (block-1);
    const bool bcurr = (block==1) && curr;

    atomic_ref<double, thread_scope_device> d_cmax(candidate_rows[t + block * NB]);

    atomic_ref<int,    thread_scope_device> d_wloc(loc_workspace[block]);
    atomic_ref<double, thread_scope_device> d_wmax(max_workspace[block]);

    //pointer to current column
    double *An = A + JJ * LDA;

    // Each block does partial reduction to find pivot row
    int    loc=-1;
    double max=0.;
    if (m < M) {
      loc = m;
      max = An[m];
    }
    block_maxloc<BLOCKSIZE>(max, loc, s_max, s_loc);

    if (t<NB) {
      // Write out to workspace
      d_cmax.store(A[loc + t * LDA], memory_order_relaxed);

      //write top row to workspace
      if (bcurr) d_ccur.store(A[0 + t*LDA], memory_order_relaxed);
    }

    if (t==0) {
      d_wloc.store(loc, memory_order_relaxed);
      d_wmax.store(max, memory_order_relaxed);
    }

    waitcnt(thread_scope_device);
    block_barrier.arrive_and_wait(memory_order_relaxed);

    //Signal the first block that the workspace is populated and wait for the swap
    if (t==0) barrier.arrive_and_wait(memory_order_relaxed);
    block_barrier.arrive_and_wait(memory_order_relaxed);

    int ii = 0;

    double Amn, Amnp1;

    for (int jj = 1; jj < JB; ++jj) {
      // Perform the row swap
      const double gmax = d_max.load(memory_order_relaxed);
      const int srcloc  = d_loc.load(memory_order_relaxed);
      const int srcrow  = d_row.load(memory_order_relaxed);
      const int srcBlock = srcloc/BLOCKSIZE + 1;

      //shift down a row
      if (bcurr) ii++;

      double acmax;
      if (t<NB) {
        //Read in the Amax row to LDS
        acmax = d_amax.load(memory_order_relaxed);
        s_Amax[t] = acmax;

        if (myrow == srcrow && block == srcBlock) {
          A[srcloc + t*LDA] = d_acur.load(memory_order_relaxed);
        }
      }
      block_barrier.arrive_and_wait(memory_order_release);

      // Scale column by max value, update next column, and record pivot candidates
      int    loc = -1;
      double max = 0.;

      if (ii <= t && m < M) {
        Amn = An[m]/gmax;
        Amnp1 = An[m + LDA] - s_Amax[jj+JJ] * Amn;

        An[m] = Amn;
        An[m + LDA] = Amnp1;

        loc = m;
        max = Amnp1;
      }

      // Each block does partial reduction to find pivot row
      block_maxloc<BLOCKSIZE>(max, loc, s_max, s_loc);

      if (t<NB) {
        amax = A[loc + t * LDA];

        if (bcurr) {
          acur = A[ii + t * LDA];
        }

        //rank-1 update of swapped rows
        if (jj+JJ < t && t < JB + JJ) {
          amax -= acmax * An[loc];

          if (bcurr) {
            acur -= acmax * An[ii];
          }
        }

        // Write out to workspace
        d_cmax.store(amax, memory_order_relaxed);

        // Write top row
        if (bcurr) d_ccur.store(acur, memory_order_relaxed);
      }

      if (t==0) {
        d_wloc.store(loc, memory_order_relaxed);
        d_wmax.store(max, memory_order_relaxed);
      }

      waitcnt(thread_scope_device);
      block_barrier.arrive_and_wait(memory_order_relaxed);

      //Signal the first block that the workspace is populated
      if (t==0) bar = barrier.arrive(memory_order_relaxed);

      //Rank 1 update while the row is exchanged on the host
      if (ii <= t && m < M) {
        for (int k=jj+1;k<JB;++k) {
          A[m + (k + JJ) * LDA] -= s_Amax[k+JJ] * Amn;
        }
      }

      //shift A one column
      An += LDA;

      if (t==0) barrier.wait(std::move(bar), memory_order_relaxed);
      block_barrier.arrive_and_wait(memory_order_relaxed);
    }

    // Perform final row swap and update
    const double gmax = d_max.load(memory_order_relaxed);
    const int srcloc  = d_loc.load(memory_order_relaxed);
    const int srcrow  = d_row.load(memory_order_relaxed);
    const int srcBlock = srcloc/BLOCKSIZE + 1;

    //shift down a row
    if (bcurr) ii++;

    if (myrow == srcrow && block == srcBlock && t<NB) {
      A[srcloc + t*LDA] = d_acur.load(memory_order_relaxed);
    }
    block_barrier.arrive_and_wait(memory_order_relaxed);

    // Scale final column by max value
    if (ii <= t && m < M) {
      An[m] /= gmax;
    }
  }
}

void HPL_pdpanrlT_device(HPL_T_panel* PANEL,
                  const int    M,
                  const int    N,
                  const int    ICOFF) {

  double *A, *L1;
  int     curr, ii, jj, lda;

  HPL_T_pmat* mat = PANEL->pmat;

  A    = PANEL->A0;
  lda  = PANEL->lda0;
  L1   = PANEL->L1;
  curr = PANEL->grid->myrow == PANEL->prow ? 1 : 0;

  HPL_T_grid* grid = PANEL->grid;
  MPI_Comm    comm = grid->col_comm;
  int myrow   = grid->myrow;
  int nprow   = grid->nprow;

  jj = ICOFF;
  if(curr != 0) {
    ii = ICOFF;
  } else {
    ii = 0;
  }

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PFACT);
#endif

  constexpr int BLOCKSIZE=1024;

  barrier<thread_scope_device> barrier(mat->barrier_space, (M+BLOCKSIZE-1)/BLOCKSIZE + 1);

  atomic_ref<int32_t, thread_scope_system> host_flag(mat->host_flag[0]);
  host_flag.store(0, memory_order_relaxed);

  if (M>0) {
    int m = M;
    int n = N;
    double *Aptr  = A + ii;
    double *L1ptr = L1 + jj * PANEL->jb;

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
    CHECK_HIP_ERROR(hipLaunchCooperativeKernel(pdfact<BLOCKSIZE>,
                                               dim3((M+BLOCKSIZE-1)/BLOCKSIZE + 1),
                                               dim3(BLOCKSIZE),
                                               params,
                                               0,
                                               stream));
  }

  int NB      = PANEL->nb;
  int icurrow = PANEL->prow;

  int cnt0 = 4 + 2 * PANEL->jb;
  double *WORK = mat->host_workspace;
  double *Wwork = WORK + cnt0;

  for (int i = 0; i < N; i++) {
    /*Wait for host_flag to update from GPU*/
    if (M>0) host_flag.wait(0, memory_order_acquire);

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_MXSWP);
#endif

    if (M>0) {
      int ilindx = static_cast<int>(WORK[1]);
      int kk     = PANEL->ii + ii + (ilindx);
      int igindx = 0;
      Mindxl2g(igindx, kk, NB, NB, myrow, 0, nprow);
      /*
       * WORK[0] := local maximum absolute value scalar,
       * WORK[1] := corresponding local  row index,
       * WORK[2] := corresponding global row index,
       * WORK[3] := coordinate of process owning this max.
       */
      WORK[2] = (double)(igindx);
      WORK[3] = (double)(myrow);

    } else {
      WORK[0] = WORK[1] = WORK[2] = HPL_rzero;
      WORK[3]                     = (double)(PANEL->grid->nprow);
    }
    HPL_all_reduce_dmxswp(WORK, cnt0, icurrow, comm, Wwork);

#ifdef HPL_DETAILED_TIMING
    HPL_ptimer(HPL_TIMING_MXSWP);
#endif

    (PANEL->ipiv)[ICOFF+i] = (int)WORK[2];

    /*Signal GPU*/
    if (M>0) host_flag.store(0, memory_order_release);
  }

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PFACT);
#endif
}

void HPL_pdrpanrlT_device(HPL_T_panel* PANEL,
                   const int    M,
                   const int    N,
                   const int    ICOFF) {

  double *A, *Aptr, *L1, *L1ptr;
  int     curr, ii, ioff, jb, jj, lda, m, n, n0, nb, nbdiv, nbmin;

  //If we're at minimum size, serial factor N columns
  if(N <= (nbmin = PANEL->algo->nbmin)) {
    HPL_pdpanrlT_device(PANEL, M, N, ICOFF);
    return;
  }

  nbdiv = PANEL->algo->nbdiv;
  ii = jj = 0;
  m       = M;
  n       = N;
  nb = jb = ((((N + nbmin - 1) / nbmin) + nbdiv - 1) / nbdiv) * nbmin;

  A     = PANEL->A0;
  lda   = PANEL->lda0;
  L1    = PANEL->L1;
  n0    = PANEL->jb;
  L1ptr = L1 + ICOFF + ICOFF * n0;
  curr  = (PANEL->grid->myrow == PANEL->prow) ? 1 : 0;

  if(curr != 0)
    Aptr = A + ICOFF + ICOFF * lda;
  else
    Aptr = A + 0 + ICOFF * lda;

  const double one  = 1.0;
  const double mone = -1.0;

  do {
    n -= jb;
    ioff = ICOFF + jj;
    /*
     * Factor current panel - Replicated solve - Local update
     */
    HPL_pdrpanrlT_device(PANEL,
                         m,
                         jb,
                         ioff);

    CHECK_ROCBLAS_ERROR(rocblas_dtrsm(handle,
                                      rocblas_side_right,
                                      rocblas_fill_upper,
                                      rocblas_operation_none,
                                      rocblas_diagonal_unit,
                                      n,
                                      jb,
                                      &one,
                                      L1ptr + jj + jj * n0,
                                      n0,
                                      L1ptr + jj + jb + jj * n0,
                                      n0));

    if(curr != 0) {
      ii += jb;
      m -= jb;
    }

    CHECK_ROCBLAS_ERROR(rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_transpose,
                                      m,
                                      n,
                                      jb,
                                      &mone,
                                      Aptr + ii + jj * lda,
                                      lda,
                                      L1ptr + jj + jb + jj * n0,
                                      n0,
                                      &one,
                                      Aptr + ii + (jj + jb) * lda,
                                      lda));

    jj += jb;
    jb = std::min(n, nb);

  } while(n > 0);
}

void HPL_pdfact_device(HPL_T_panel* PANEL) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdfact recursively factorizes a  1-dimensional  panel of columns.
   * The  RPFACT  function pointer specifies the recursive algorithm to be
   * used, either Crout, Left- or Right looking.  NBMIN allows to vary the
   * recursive stopping criterium in terms of the number of columns in the
   * panel, and  NDIV allows to specify the number of subpanels each panel
   * should be divided into. Usuallly a value of 2 will be chosen. Finally
   * PFACT is a function pointer specifying the non-recursive algorithm to
   * to be used on at most NBMIN columns. One can also choose here between
   * Crout, Left- or Right looking.  Empirical tests seem to indicate that
   * values of 4 or 8 for NBMIN give the best results.
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
   * words, and  gam2-3  is  an estimate of the  Level 2 and Level 3  BLAS
   * rate of execution. The  recursive  algorithm  allows indeed to almost
   * achieve  Level 3 BLAS  performance  in the panel factorization.  On a
   * large  number of modern machines,  this  operation is however latency
   * bound,  meaning  that its cost can  be estimated  by only the latency
   * portion N0 * log_2(P) * lat.  Mono-directional links will double this
   * communication cost.
   *
   * Arguments
   * =========
   *
   * PANEL   (local input/output)          HPL_T_panel *
   *         On entry,  PANEL  points to the data structure containing the
   *         panel information.
   *
   * ---------------------------------------------------------------------
   */

  int curr = PANEL->grid->myrow == PANEL->prow ? 1 : 0;

  HPL_pdrpanrlT_device(PANEL, PANEL->mp, PANEL->jb, 0);

  if (curr) {
    HPL_dlatcpy_gpu(PANEL->jb, PANEL->jb, PANEL->L1, PANEL->jb, PANEL->A, PANEL->lda);
  }
}

