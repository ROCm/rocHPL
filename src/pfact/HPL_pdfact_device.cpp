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


__host__ __device__ constexpr int prev_pow2(const int N) {
  return (1 << (31 - __builtin_clz(N-1)));
}

__device__ inline void maxloc_block_reduce(int &loc,
                                           double &max,
                                           int *s_loc,
                                           double *s_max) {
  const int t = threadIdx.x;

  s_loc[t] = loc;
  s_max[t] = max;

  __syncthreads();

  int active = prev_pow2(blockDim.x);

  if (t<active && t+active < blockDim.x) {
    if (std::abs(s_max[t+active]) > std::abs(s_max[t])) {
      s_max[t] = s_max[t+active];
      s_loc[t] = s_loc[t+active];
    }
  }
  __syncthreads();

  active >>= 1;
  for (; active>0;active>>=1) {
    if (t<active) {
      if (std::abs(s_max[t+active]) > std::abs(s_max[t])) {
        s_max[t] = s_max[t+active];
        s_loc[t] = s_loc[t+active];
      }
    }
    __syncthreads();
  }
}

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
                       int32_t  *__restrict__ host_flag,
                       double   *__restrict__ host_workspace,
                       int32_t  *__restrict__ locks) {

  const int t = threadIdx.x;
  const int block = blockIdx.x;
  const int m = t + NB * block;

  extern __shared__ char s_array[];

  double *s_Amax = reinterpret_cast<double*>(&s_array[0]);
  double *s_max  = reinterpret_cast<double*>(&s_array[blockDim.x * sizeof(double)]);
  int    *s_loc  = reinterpret_cast<int*>(&s_array[2 * blockDim.x * sizeof(double)]);

  int l=0;

  //Device workspaces to hold pivoted rows at agent scope
  double *Amax = dev_workspace + 4;
  double *Acur = dev_workspace + 4 + NB;
  double *candidate_rows = dev_workspace + 4 + 2 * NB;

  //pointer to current column
  double *An = A + JJ * LDA;

  // Each block does partial reduction to find pivot row
  int    loc=-1;
  double max=0.;
  if (t + NB * block < M) {
    loc = t + NB * block;
    max = An[t + NB * block];
  }
  maxloc_block_reduce(loc, max, s_loc, s_max);

  loc = s_loc[0];

  // Write out to workspace
  __hip_atomic_store(&candidate_rows[t + block * NB], A[loc + t * LDA], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  __syncthreads();

  if (t==0) {
    __hip_atomic_store(&loc_workspace[block], loc,      __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&max_workspace[block], s_max[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    //Signal the first block that the workspace is populated
    atomicAdd(locks+l, 1);
  }

  if (block==0) {
    if (t==0) {
      //Wait for all blocks to have completed partial reduction
      while(__hip_atomic_load(locks+l, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != gridDim.x) {}
    }
    __syncthreads();

    // Complete maxloc reduction
    loc=-1;
    max=0.;
    for (int id=t;id<gridDim.x;id+=blockDim.x) {
      const double r_max = __hip_atomic_load(&max_workspace[id], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (std::abs(r_max) > std::abs(max)) {
        loc = id;
        max = r_max;
      }
    }
    maxloc_block_reduce(loc, max, s_loc, s_max);

    loc = s_loc[0]; //block number where local max row resides
    max = s_max[0];

    //Read local candidate pivot row out of agent work buffer
    const double almax = __hip_atomic_load(&candidate_rows[t + loc * NB], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    loc = __hip_atomic_load(&loc_workspace[loc], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    //write row into host swap space
    __hip_atomic_store(&host_workspace[4+t],      almax,          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

    if (curr) {
      __hip_atomic_store(&host_workspace[4+t + NB], A[0   + t*LDA], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    __syncthreads();

    if(t==0) {
      //Write row number and max val to header
      __hip_atomic_store(&host_workspace[0], static_cast<double>(max), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(&host_workspace[1], static_cast<double>(loc), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      //Mark pivot row as found
      __hip_atomic_store(host_flag, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

      //Wait for Host to unlock
      while (__hip_atomic_load(host_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0) {}
    }

    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

    //populate the max row device workspace. Read system scope -> write agent scope
    const double amax = __hip_atomic_load(&host_workspace[4+t],    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    const double acur = __hip_atomic_load(&host_workspace[4+t+NB], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

    __hip_atomic_store(&Amax[t], amax, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&Acur[t], acur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __syncthreads();

    if (t==0) {
      const double maxval = __hip_atomic_load(&host_workspace[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const double maxloc = __hip_atomic_load(&host_workspace[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const double maxrow = __hip_atomic_load(&host_workspace[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      __hip_atomic_store(&dev_workspace[0], maxval, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&dev_workspace[1], maxloc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&dev_workspace[3], maxrow, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      //Unlock other blocks
      __hip_atomic_store(locks+l, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    //Wait for block 0 to complete swap
    if (t==0) {
      while(__hip_atomic_load(locks+l, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) {}
    }
  }
  __syncthreads();

  int ii = 0;

  double Amn, Amnp1;

  for (int jj = 1; jj < JB; ++jj) {
    //Cycle lock
    l = (l+1)%2;

    // Perform the row swap
    const int srcloc = static_cast<int>(__hip_atomic_load(&dev_workspace[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    const int srcrow = static_cast<int>(__hip_atomic_load(&dev_workspace[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
    const int srcBlock = srcloc/NB;

    //Read in the Amax row to LDS
    const double amax = __hip_atomic_load(&Amax[t], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    s_Amax[t] = amax;

    const double gmax = __hip_atomic_load(&dev_workspace[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    if (block==0) {
      L1[t + (jj-1)*NB] = amax;

      //shift down a row
      if (curr) ii++;
    }
    if (myrow == srcrow && block == srcBlock) {
      A[srcloc + t*LDA] = __hip_atomic_load(&Acur[t], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    __syncthreads();

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
    maxloc_block_reduce(loc, max, s_loc, s_max);

    loc = s_loc[0];
    max = s_max[0];

    // Write out to workspace
    __hip_atomic_store(&candidate_rows[t + block * NB], A[loc + t * LDA], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __syncthreads();

    if (t==0) {
      __hip_atomic_store(&loc_workspace[block], loc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&max_workspace[block], max, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      //Signal the first block that the workspace is populated
      atomicAdd(locks+l, 1);
    }

    if (block==0) {
      if (t==0) {
        //Wait for all blocks to have completed partial reduction
        while(__hip_atomic_load(locks+l, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != gridDim.x) {}
      }
      __syncthreads();

      // Complete maxloc reduction
      loc=-1;
      max=0.;
      for (int id=t;id<gridDim.x;id+=blockDim.x) {
        const double r_max = __hip_atomic_load(&max_workspace[id], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (std::abs(r_max) > std::abs(max)) {
          loc = id;
          max = r_max;
        }
      }
      maxloc_block_reduce(loc, max, s_loc, s_max);

      loc = s_loc[0]; //block number where local max row resides
      max = s_max[0];

      //Read local candidate pivot row out of agent work buffer
      const double almax = __hip_atomic_load(&candidate_rows[t + loc * NB], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      loc = __hip_atomic_load(&loc_workspace[loc], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      //write row into host swap space
      __hip_atomic_store(&host_workspace[4+t],      almax, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      if (curr) {
        __hip_atomic_store(&host_workspace[4+t + NB], A[ii + t*LDA],  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }

      __syncthreads();

      if(t==0) {
        __hip_atomic_store(&host_workspace[0], static_cast<double>(max), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&host_workspace[1], static_cast<double>(loc), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

        //Mark pivot row as found
        __hip_atomic_store(host_flag, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }

    //Rank 1 update while the row is exchanged on the host
    if (ii <= t && m < M) {
      for (int k=jj+1;k<JB;++k) {
        A[m + (k + JJ) * LDA] -= s_Amax[k+JJ] * Amn;
      }
    }

    if (block==0) {
      if(t==0) {
        //Wait for Host to unlock
        while (__hip_atomic_load(host_flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0) {}
      }
      __syncthreads();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

      //populate the max row device workspace. Read system scope -> write agent scope
      double amax = __hip_atomic_load(&host_workspace[4+t],    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      double acur = __hip_atomic_load(&host_workspace[4+t+NB], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      const double amaxn = __hip_atomic_load(&host_workspace[4+jj+JJ-1],    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      const double acurn = __hip_atomic_load(&host_workspace[4+jj+JJ-1+NB], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

      //rank-1 update of swapped rows
      if (jj+JJ < t && t < JB + JJ) {
        amax -= s_Amax[t] * amaxn;
        acur -= s_Amax[t] * acurn;
      }

      __hip_atomic_store(&Amax[t], amax, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&Acur[t], acur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      __syncthreads();

      if (t==0) {
        const double maxval = __hip_atomic_load(&host_workspace[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const double maxloc = __hip_atomic_load(&host_workspace[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const double maxrow = __hip_atomic_load(&host_workspace[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(&dev_workspace[0], maxval, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&dev_workspace[1], maxloc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&dev_workspace[3], maxrow, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

        //Unlock other blocks
        __hip_atomic_store(locks+l, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      //Wait for block 0 to complete swap
      if (t==0) {
        while(__hip_atomic_load(locks+l, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) {}
      }
    }
    __syncthreads();

    //shift A one column
    An += LDA;
  }

  // Perform final row swap and update
  const double gmax = __hip_atomic_load(&dev_workspace[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const int srcloc = static_cast<int>(__hip_atomic_load(&dev_workspace[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  const int srcrow = static_cast<int>(__hip_atomic_load(&dev_workspace[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  const int srcBlock = srcloc/NB;

  if (block==0) {
    L1[t + (JB-1)*NB] = __hip_atomic_load(&Amax[t], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    //shift down a row
    if (curr) ii++;
  }
  if (myrow == srcrow && block == srcBlock) {
    A[srcloc + t*LDA] = __hip_atomic_load(&Acur[t], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  __syncthreads();

  // Scale final column by max value
  if (ii <= t && m < M) {
    An[m] /= gmax;
  }
}

void HPL_pdpanrlT_device(HPL_T_panel* PANEL,
                  const int    M,
                  const int    N,
                  const int    ICOFF) {

  double *A, *L1;
  int     curr, ii, jj, lda;

  A    = PANEL->A;
  lda  = PANEL->lda;
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

  if (M>0) {
    pdfact<<<dim3((M+PANEL->jb-1)/PANEL->jb),
             dim3(PANEL->jb),
             (2 * sizeof(double) + sizeof(int)) * PANEL->jb,
             stream>>>(M,
                       PANEL->jb,
                       N,
                       A + ii,
                       lda,
                       curr,
                       myrow,
                       jj,
                       L1 + jj * PANEL->jb,
                       PANEL->loc_workspace,
                       PANEL->max_workspace,
                       PANEL->dev_workspace,
                       PANEL->host_flag,
                       PANEL->host_workspace,
                       PANEL->locks);
  }

  int NB      = PANEL->nb;
  int icurrow = PANEL->prow;

  int cnt0 = 4 + 2 * PANEL->jb;
  double *WORK = PANEL->host_workspace;
  double *Wwork = WORK + cnt0;

  for (int i = 0; i < N; i++) {
    if (M>0) {
      /*Wait for host_flag to update from GPU*/
      while (__atomic_load_n(PANEL->host_flag, __ATOMIC_ACQUIRE) != 1) { }
    }

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

    if (M>0) {
      /*Signal GPU*/
      __atomic_store_n(PANEL->host_flag, 0, __ATOMIC_RELEASE);
    }
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

  A     = PANEL->A;
  lda   = PANEL->lda;
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

