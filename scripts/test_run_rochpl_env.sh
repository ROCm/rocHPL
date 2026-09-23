#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test_dir=$(mktemp -d)
trap '[[ -n ${test_dir:-} && -d $test_dir ]] && rm -rf -- "$test_dir"' EXIT

mkdir -p "$test_dir/bin" "$test_dir/prefix/bin" "$test_dir/rocm/bin" "$test_dir/rocblas"
printf 'ID=ubuntu\n' > "$test_dir/os-release"
printf '%s\n' '#!/usr/bin/env bash' 'printf "STUB device=%s args=%s\n" "${ROCR_VISIBLE_DEVICES:-unset}" "$*"' > "$test_dir/prefix/bin/rochpl"
printf '%s\n' '#!/usr/bin/env bash' 'printf "gpu,node\n0,0\n1,0\n2,0\n3,0\n"' > "$test_dir/rocm/bin/amd-smi"
printf '%s\n' '#!/usr/bin/env bash' 'exit 1' > "$test_dir/bin/numactl"
chmod +x "$test_dir/prefix/bin/rochpl" "$test_dir/rocm/bin/amd-smi" "$test_dir/bin/numactl"

sed -e "s|@CMAKE_INSTALL_PREFIX@|$test_dir/prefix|g" \
    -e "s|@ROCM_PATH@|$test_dir/rocm|g" \
    -e "s|@ROCBLAS_LIB_PATH@|$test_dir/rocblas|g" \
    -e "s|/etc/os-release|$test_dir/os-release|g" \
    "$script_dir/run_rochpl.in" > "$test_dir/run_rochpl"
chmod +x "$test_dir/run_rochpl"

run_ok() {
  local expected_local=$1 expect_binding=$2 name=$3 grid_p=$4 grid_q=$5
  shift 5
  local output
  if ! output=$(env -i PATH="$test_dir/bin:$PATH" LD_LIBRARY_PATH= "$@" \
      "$test_dir/run_rochpl" -P "$grid_p" -Q "$grid_q" -p 1 -q 4 \
      -N 64 --NB 64 --devices 0,1,2,3 2>&1); then
    printf 'FAIL %s: %s\n' "$name" "$output" >&2
    exit 1
  fi
  if [[ $output != *"STUB device=$expected_local "* ]]; then
    printf 'FAIL %s: wrong GPU mapping: %s\n' "$name" "$output" >&2
    exit 1
  fi
  if [[ $expect_binding == yes && $output != *"Node Binding: Process $expected_local "* ]]; then
    printf 'FAIL %s: missing local rank binding: %s\n' "$name" "$output" >&2
    exit 1
  fi
  printf 'PASS %s\n' "$name"
}

run_fail() {
  local name=$1 expected_error=$2
  shift 2
  local output
  if output=$(env -i PATH="$test_dir/bin:$PATH" LD_LIBRARY_PATH= "$@" \
      "$test_dir/run_rochpl" -P 1 -Q 4 -p 1 -q 4 \
      -N 64 --NB 64 --devices 0,1,2,3 2>&1); then
    printf 'FAIL %s: expected failure\n' "$name" >&2
    exit 1
  fi
  if [[ $output != *"$expected_error"* ]]; then
    printf 'FAIL %s: unexpected error: %s\n' "$name" "$output" >&2
    exit 1
  fi
  printf 'PASS %s\n' "$name"
}

run_ok 2 yes native-slurm 1 4 \
  SLURM_PROCID=2 SLURM_NTASKS=4 SLURM_LOCALID=2 'SLURM_TASKS_PER_NODE=4(x1)'
run_ok 2 yes matching-slurm-pmi 1 4 \
  SLURM_PROCID=2 SLURM_NTASKS=4 SLURM_LOCALID=2 SLURM_TASKS_PER_NODE=4 \
  PMI_RANK=2 PMI_SIZE=4 MPI_LOCALRANKID=2 MPI_LOCALNRANKS=4
for rank in 0 1 2 3; do
  run_ok "$rank" yes "nested-intel-mpi-$rank" 1 4 \
    SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 SLURM_TASKS_PER_NODE=1 \
    "PMI_RANK=$rank" PMI_SIZE=4 "MPI_LOCALRANKID=$rank" MPI_LOCALNRANKS=4
done
run_ok 1 yes open-mpi-priority 1 4 \
  OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_SIZE=4 \
  OMPI_COMM_WORLD_LOCAL_RANK=1 OMPI_COMM_WORLD_LOCAL_SIZE=4 \
  SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 SLURM_TASKS_PER_NODE=1 \
  PMI_RANK=2 PMI_SIZE=4 MPI_LOCALRANKID=2 MPI_LOCALNRANKS=4
run_ok 3 yes flux 1 4 \
  FLUX_TASK_RANK=3 FLUX_JOB_SIZE=4 FLUX_JOB_NNODES=1 FLUX_TASK_LOCAL_ID=3
run_ok 1 yes mvapich2 1 4 \
  MV2_COMM_WORLD_RANK=1 MV2_COMM_WORLD_SIZE=4 \
  MV2_COMM_WORLD_LOCAL_RANK=1 MV2_COMM_WORLD_LOCAL_SIZE=4
run_ok 3 yes nested-mvapich2 1 4 \
  SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 SLURM_TASKS_PER_NODE=1 \
  MV2_COMM_WORLD_RANK=3 MV2_COMM_WORLD_SIZE=4 \
  MV2_COMM_WORLD_LOCAL_RANK=3 MV2_COMM_WORLD_LOCAL_SIZE=4
run_ok 2 yes standalone-pmi 1 4 \
  PMI_RANK=2 PMI_SIZE=4 MPI_LOCALRANKID=2 MPI_LOCALNRANKS=4
run_ok 0 no second-node-pmi 2 4 \
  SLURM_PROCID=0 SLURM_NTASKS=2 SLURM_LOCALID=0 SLURM_TASKS_PER_NODE=1 \
  PMI_RANK=4 PMI_SIZE=8 MPI_LOCALRANKID=0 MPI_LOCALNRANKS=4
run_ok 2 yes incomplete-pmi-with-valid-slurm 1 4 \
  SLURM_PROCID=2 SLURM_NTASKS=4 SLURM_LOCALID=2 SLURM_TASKS_PER_NODE=4 \
  PMI_RANK=2 PMI_SIZE=4 MPI_LOCALNRANKS=4
run_fail invalid-pmi 'Unable to determine valid MPI rank environment' \
  PMI_RANK=invalid PMI_SIZE=4 MPI_LOCALRANKID=0 MPI_LOCALNRANKS=4
run_fail incomplete-pmi 'Unable to determine valid MPI rank environment' \
  PMI_RANK=0 PMI_SIZE=4 MPI_LOCALRANKID=0
run_fail invalid-slurm 'Unable to determine valid Slurm rank environment' \
  SLURM_PROCID=0 SLURM_NTASKS=4 SLURM_LOCALID=4 SLURM_TASKS_PER_NODE=4
