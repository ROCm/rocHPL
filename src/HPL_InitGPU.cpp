/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2022 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#include "hpl.hpp"
#include <algorithm>

rocblas_handle handle;

hipStream_t computeStream, dataStream;

hipEvent_t swapStartEvent[HPL_N_UPD], update[HPL_N_UPD];
hipEvent_t dgemmStart[HPL_N_UPD], dgemmStop[HPL_N_UPD];

static char host_name[MPI_MAX_PROCESSOR_NAME];

/*
  This function finds out how many MPI processes are running on the same node
  and assigns a local rank that can be used to map a process to a device.
  This function needs to be called by all the MPI processes.
*/
void HPL_InitGPU(const HPL_T_grid* GRID) {
  char host_name[MPI_MAX_PROCESSOR_NAME];

  int i, n, namelen, rank, nprocs;
  int dev;

  int nprow, npcol, myrow, mycol;
  (void)HPL_grid_info(GRID, &nprow, &npcol, &myrow, &mycol);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(host_name, &namelen);

  int localRank = GRID->local_mycol + GRID->local_myrow * GRID->local_npcol;
  int localSize = GRID->local_npcol * GRID->local_nprow;

  /* Find out how many GPUs are in the system and their device number */
  int deviceCount;
  hipGetDeviceCount(&deviceCount);

  if(deviceCount < 1) {
    if(localRank == 0)
      HPL_pwarn(stderr,
                __LINE__,
                "HPL_InitGPU",
                "Node %s found no GPUs. Is the ROCm kernel module loaded?",
                host_name);
    MPI_Finalize();
    exit(1);
  }

  dev = localRank % deviceCount;

#ifdef HPL_VERBOSE_PRINT
  if(rank < localSize) {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, dev);

    printf("GPU  Binding: Process %d [(p,q)=(%d,%d)] GPU: %d, pciBusID %x \n",
           rank,
           GRID->local_myrow,
           GRID->local_mycol,
           dev,
           props.pciBusID);
  }
#endif

  /* Assign device to MPI process, initialize BLAS and probe device properties
   */
  hipSetDevice(dev);

  hipStreamCreate(&computeStream);
  hipStreamCreate(&dataStream);

  hipEventCreate(swapStartEvent + HPL_LOOK_AHEAD);
  hipEventCreate(swapStartEvent + HPL_UPD_1);
  hipEventCreate(swapStartEvent + HPL_UPD_2);

  hipEventCreate(update + HPL_LOOK_AHEAD);
  hipEventCreate(update + HPL_UPD_1);
  hipEventCreate(update + HPL_UPD_2);

  hipEventCreate(dgemmStart + HPL_LOOK_AHEAD);
  hipEventCreate(dgemmStart + HPL_UPD_1);
  hipEventCreate(dgemmStart + HPL_UPD_2);

  hipEventCreate(dgemmStop + HPL_LOOK_AHEAD);
  hipEventCreate(dgemmStop + HPL_UPD_1);
  hipEventCreate(dgemmStop + HPL_UPD_2);
}

void HPL_FreeGPU() {
  hipEventDestroy(swapStartEvent[HPL_LOOK_AHEAD]);
  hipEventDestroy(swapStartEvent[HPL_UPD_1]);
  hipEventDestroy(swapStartEvent[HPL_UPD_2]);

  hipEventDestroy(update[HPL_LOOK_AHEAD]);
  hipEventDestroy(update[HPL_UPD_1]);
  hipEventDestroy(update[HPL_UPD_2]);

  hipEventDestroy(dgemmStart[HPL_LOOK_AHEAD]);
  hipEventDestroy(dgemmStart[HPL_UPD_1]);
  hipEventDestroy(dgemmStart[HPL_UPD_2]);

  hipEventDestroy(dgemmStop[HPL_LOOK_AHEAD]);
  hipEventDestroy(dgemmStop[HPL_UPD_1]);
  hipEventDestroy(dgemmStop[HPL_UPD_2]);

  hipStreamDestroy(dataStream);
  hipStreamDestroy(computeStream);
}
