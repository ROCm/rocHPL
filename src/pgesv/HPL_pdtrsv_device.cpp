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

#define BLOCK_SIZE 512
__global__ void setZero(const int N, double* __restrict__ X) {
  const int    t  = threadIdx.x;
  const int    b  = blockIdx.x;
  const size_t id = b * BLOCK_SIZE + t; // row id

  if(id < N) { X[id] = 0.0; }
}

void HPL_pdtrsv(HPL_T_grid* GRID, HPL_T_pmat* AMAT) {
  /*
   * Purpose
   * =======
   *
   * HPL_pdtrsv solves an upper triangular system of linear equations.
   *
   * The rhs is the last column of the N by N+1 matrix A. The solve starts
   * in the process  column owning the  Nth  column of A, so the rhs b may
   * need to be moved one process column to the left at the beginning. The
   * routine therefore needs  a column  vector in every process column but
   * the one owning  b. The result is  replicated in all process rows, and
   * returned in XR, i.e. XR is of size nq = LOCq( N ) in all processes.
   *
   * The algorithm uses decreasing one-ring broadcast in process rows  and
   * columns  implemented  in terms of  synchronous communication point to
   * point primitives.  The  lookahead of depth 1 is used to minimize  the
   * critical path. This entire operation is essentially ``latency'' bound
   * and an estimate of its running time is given by:
   *
   *    (move rhs) lat + N / ( P bdwth ) +
   *    (solve)    ((N / NB)-1) 2 (lat + NB / bdwth) +
   *               gam2 N^2 / ( P Q ),
   *
   * where  gam2   is an estimate of the   Level 2 BLAS rate of execution.
   * There are  N / NB  diagonal blocks. One must exchange  2  messages of
   * length NB to compute the next  NB  entries of the vector solution, as
   * well as performing a total of N^2 floating point operations.
   *
   * Arguments
   * =========
   *
   * GRID    (local input)                 HPL_T_grid *
   *         On entry,  GRID  points  to the data structure containing the
   *         process grid information.
   *
   * AMAT    (local input/output)          HPL_T_pmat *
   *         On entry,  AMAT  points  to the data structure containing the
   *         local array information.
   *
   * ---------------------------------------------------------------------
   */

  MPI_Comm Ccomm, Rcomm;
  double * Aprev = NULL, *XC = NULL, *XR = NULL, *Xd = NULL, *Xdprev = NULL,
         *W  = NULL;
  double *dA = NULL, *dAprev = NULL, *dAptr, *dXC = NULL, *dXR = NULL,
         *dXd = NULL, *dXdprev = NULL, *dW = NULL;
  int Alcol, Alrow, Anpprev, Anp, Anq, Bcol, Cmsgid, GridIsNotPx1, GridIsNot1xQ,
      Rmsgid, colprev, kb, kbprev, lda, mycol, myrow, n, n1, n1p,
      n1pprev = 0, nb, npcol, nprow, rowprev, tmp1, tmp2;
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PTRSV);
#endif
  if((n = AMAT->n) <= 0) return;

  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);
  Rcomm        = GRID->row_comm;
  Rmsgid       = MSGID_BEGIN_PTRSV;
  Ccomm        = GRID->col_comm;
  Cmsgid       = MSGID_BEGIN_PTRSV + 1;
  GridIsNot1xQ = (nprow > 1);
  GridIsNotPx1 = (npcol > 1);

  nb  = AMAT->nb;
  lda = AMAT->ld;

  Mnumroc(Anp, n, nb, nb, myrow, 0, nprow);
  Mnumroc(Anq, n, nb, nb, mycol, 0, npcol);

  dA  = AMAT->dA;
  dXR = AMAT->dX;
  XR  = AMAT->W + 2 * Anp;

  XC  = AMAT->W;
  dXC = AMAT->dW;

  W  = AMAT->W + Anp;
  dW = AMAT->dW + Anp;

  hipStream_t stream;
  CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

  /*
   * Move the rhs in the process column owning the last column of A.
   */

  tmp1  = (n - 1) / nb;
  Alrow = tmp1 - (tmp1 / nprow) * nprow;
  Alcol = tmp1 - (tmp1 / npcol) * npcol;
  kb    = n - tmp1 * nb;

  dAptr      = (double*)(dA);
  double* dB = Mptr(dAptr, 0, Anq, lda);

  Mindxg2p(n, nb, nb, Bcol, 0, npcol);

  if(Anp > 0) {
    if(Alcol != Bcol) {
      if(mycol == Bcol) {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            dXC, dB, Anp * sizeof(double), hipMemcpyDeviceToDevice, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        (void)HPL_send(dXC, Anp, Alcol, Rmsgid, Rcomm);
      } else if(mycol == Alcol) {
        (void)HPL_recv(dXC, Anp, Bcol, Rmsgid, Rcomm);
      }
    } else {
      if(mycol == Bcol) {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            dXC, dB, Anp * sizeof(double), hipMemcpyDeviceToDevice, stream));
      }
    }
  }

  Rmsgid = (Rmsgid + 2 > MSGID_END_PTRSV ? MSGID_BEGIN_PTRSV : Rmsgid + 2);
  if(mycol != Alcol) {
    if(Anp) {
      size_t grid_size = (Anp + BLOCK_SIZE - 1) / BLOCK_SIZE;
      setZero<<<grid_size, BLOCK_SIZE, 0, stream>>>(Anp, dXC);
      CHECK_HIP_ERROR(hipGetLastError());
    }
  }
  /*
   * Set up lookahead
   */
  n1 = (npcol - 1) * nb;
  n1 = Mmax(n1, nb);

  Anpprev = Anp;
  dAprev = dAptr = Mptr(dAptr, 0, Anq, lda);
  Xdprev         = XR;
  dXdprev        = dXR;
  tmp1           = n - kb;
  tmp1 -= (tmp2 = Mmin(tmp1, n1));
  MnumrocI(n1pprev, tmp2, Mmax(0, tmp1), nb, nb, myrow, 0, nprow);

  if(myrow == Alrow) { Anpprev = (Anp -= kb); }
  if(mycol == Alcol) {
    dAprev = (dAptr -= lda * kb);
    Anq -= kb;
    Xdprev  = (Xd = XR + Anq);
    dXdprev = (dXd = dXR + Anq);
    if(myrow == Alrow) {
      CHECK_ROCBLAS_ERROR(rocblas_dtrsv(handle,
                                        rocblas_fill_upper,
                                        rocblas_operation_none,
                                        rocblas_diagonal_non_unit,
                                        kb,
                                        dAptr + Anp,
                                        lda,
                                        dXC + Anp,
                                        1));
      CHECK_ROCBLAS_ERROR(rocblas_dcopy(handle, kb, dXC + Anp, 1, dXd, 1));
    }
  }

  rowprev = Alrow;
  Alrow   = MModSub1(Alrow, nprow);
  colprev = Alcol;
  Alcol   = MModSub1(Alcol, npcol);
  kbprev  = kb;
  n -= kb;
  tmp1 = n - (kb = nb);
  tmp1 -= (tmp2 = Mmin(tmp1, n1));
  MnumrocI(n1p, tmp2, Mmax(0, tmp1), nb, nb, myrow, 0, nprow);
  /*
   * Start the operations
   */
  while(n > 0) {
    if(mycol == Alcol) {
      dAptr -= lda * kb;
      Anq -= kb;
      Xd  = XR + Anq;
      dXd = dXR + Anq;
    }
    if(myrow == Alrow) { Anp -= kb; }
    /*
     * Broadcast  (decreasing-ring)  of  previous solution block in previous
     * process column,  compute  partial update of current block and send it
     * to current process column.
     */
    if(mycol == colprev) {
      /*
       * Send previous solution block in process row above
       */
      if(myrow == rowprev) {
        if(GridIsNot1xQ) {
          if(kbprev) {
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            (void)HPL_send(
                dXdprev, kbprev, MModSub1(myrow, nprow), Cmsgid, Ccomm);
          }
        }
      } else {
        if(kbprev) {
          (void)HPL_recv(
              dXdprev, kbprev, MModAdd1(myrow, nprow), Cmsgid, Ccomm);
        }
      }
      /*
       * Compute partial update of previous solution block and send it to cur-
       * rent column
       */
      if(n1pprev > 0) {
        tmp1              = Anpprev - n1pprev;
        const double one  = 1.0;
        const double mone = -1.0;
        CHECK_ROCBLAS_ERROR(rocblas_dgemv(handle,
                                          rocblas_operation_none,
                                          n1pprev,
                                          kbprev,
                                          &mone,
                                          dAprev + tmp1,
                                          lda,
                                          dXdprev,
                                          1,
                                          &one,
                                          dXC + tmp1,
                                          1));
        if(GridIsNotPx1) {
          if(n1pprev) {
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            (void)HPL_send(dXC + tmp1, n1pprev, Alcol, Rmsgid, Rcomm);
          }
        }
      }
      /*
       * Finish  the (decreasing-ring) broadcast of the solution block in pre-
       * vious process column
       */
      if((myrow != rowprev) && (myrow != MModAdd1(rowprev, nprow))) {
        if(kbprev) {
          CHECK_HIP_ERROR(hipDeviceSynchronize());
          (void)HPL_send(
              dXdprev, kbprev, MModSub1(myrow, nprow), Cmsgid, Ccomm);
        }
      }
    } else if(mycol == Alcol) {
      /*
       * Current  column  receives  and accumulates partial update of previous
       * solution block
       */
      if(n1pprev > 0) {
        if(n1pprev) {
          (void)HPL_recv(dW, n1pprev, colprev, Rmsgid, Rcomm);
          const double one = 1.0;
          CHECK_ROCBLAS_ERROR(rocblas_daxpy(
              handle, n1pprev, &one, dW, 1, dXC + Anpprev - n1pprev, 1));
        }
      }
    }
    /*
     * Solve current diagonal block
     */
    if((mycol == Alcol) && (myrow == Alrow)) {
      CHECK_ROCBLAS_ERROR(rocblas_dtrsv(handle,
                                        rocblas_fill_upper,
                                        rocblas_operation_none,
                                        rocblas_diagonal_non_unit,
                                        kb,
                                        dAptr + Anp,
                                        lda,
                                        dXC + Anp,
                                        1));
      CHECK_ROCBLAS_ERROR(
          rocblas_dcopy(handle, kb, dXC + Anp, 1, dXR + Anq, 1));
    }
    /*
     *  Finish previous update
     */
    if((mycol == colprev) && ((tmp1 = Anpprev - n1pprev) > 0)) {
      const double one  = 1.0;
      const double mone = -1.0;
      CHECK_ROCBLAS_ERROR(rocblas_dgemv(handle,
                                        rocblas_operation_none,
                                        tmp1,
                                        kbprev,
                                        &mone,
                                        dAprev,
                                        lda,
                                        dXdprev,
                                        1,
                                        &one,
                                        dXC,
                                        1));
    }
    /*
     *  Save info of current step and update info for the next step
     */
    if(mycol == Alcol) {
      dAprev  = dAptr;
      Xdprev  = Xd;
      dXdprev = dXd;
    }
    if(myrow == Alrow) { Anpprev -= kb; }

    rowprev = Alrow;
    colprev = Alcol;
    n1pprev = n1p;
    kbprev  = kb;
    n -= kb;
    Alrow = MModSub1(Alrow, nprow);
    Alcol = MModSub1(Alcol, npcol);
    tmp1  = n - (kb = nb);
    tmp1 -= (tmp2 = Mmin(tmp1, n1));
    MnumrocI(n1p, tmp2, Mmax(0, tmp1), nb, nb, myrow, 0, nprow);

    Rmsgid = (Rmsgid + 2 > MSGID_END_PTRSV ? MSGID_BEGIN_PTRSV : Rmsgid + 2);
    Cmsgid =
        (Cmsgid + 2 > MSGID_END_PTRSV ? MSGID_BEGIN_PTRSV + 1 : Cmsgid + 2);
  }
  /*
   * Replicate last solution block
   */
  if(mycol == colprev) {
    if(kbprev) {
      CHECK_HIP_ERROR(hipDeviceSynchronize());
      (void)HPL_broadcast((void*)(dXR), kbprev, HPL_DOUBLE, rowprev, Ccomm);
    }
  }

#ifdef HPL_DETAILED_TIMING
  HPL_ptimer(HPL_TIMING_PTRSV);
#endif
}
