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
  CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));

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
    CHECK_HIP_ERROR(hipGetDeviceProperties(&props, dev));

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
  CHECK_HIP_ERROR(hipSetDevice(dev));

  CHECK_HIP_ERROR(hipStreamCreate(&computeStream));
  CHECK_HIP_ERROR(hipStreamCreate(&dataStream));

  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(swapStartEvent + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(update + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(update + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(update + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStart + HPL_UPD_2));

  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_LOOK_AHEAD));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_UPD_1));
  CHECK_HIP_ERROR(hipEventCreate(dgemmStop + HPL_UPD_2));
}

void HPL_FreeGPU() {
  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(swapStartEvent[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(update[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStart[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_LOOK_AHEAD]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_UPD_1]));
  CHECK_HIP_ERROR(hipEventDestroy(dgemmStop[HPL_UPD_2]));

  CHECK_HIP_ERROR(hipStreamDestroy(dataStream));
  CHECK_HIP_ERROR(hipStreamDestroy(computeStream));
}
