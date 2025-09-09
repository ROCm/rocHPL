#!/bin/bash

#CRAY_CPE_VERSION=24.11
#CRAY_CPE_ROCM_DEFAULT_VERSION=6.2.4
#CRAY_CPE_ACCEL_SUPPORT=1


#source /opt/cray/pe/cpe/$CRAY_CPE_VERSION/restore_lmod_system_defaults.sh
#module reset
#source /ccs/proj/gen010/tools/install/scorep-modules
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.2.4-papi-7.2.0b1/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.1.3-papi-7.2.0b1/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.2.4-papi-e-master-9e1f97a4979257a7fe4f4f04f842c1d23318a60b/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-2024.06.rocprof_sdk-adanalis/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-dev-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-master/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-dev-debug-PrgEnv-amd-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-c54046bc2c2a54ce97333d7f78900875d12b13f7/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.2-papi-2024.06.rocprof_sdk-adanalis/scorep-modules.sh
#export PAPI_ROCP_SDK_ROOT="/lustre/orion/proj-shared/gen010/rocprofiler-sdk-6.3.2"
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-dev-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-master/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-dev-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-c54046bc2c2a54ce97333d7f78900875d12b13f7/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-dev-debug-PrgEnv-amd-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-c54046bc2c2a54ce97333d7f78900875d12b13f7/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-17ded13a78c26988d3e8ff72daa130124aa02ed8/scorep-modules.sh
#source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-cray-8.6.0-cpe-24.11-rocm-6.3.1-papi-7.1.0.4/scorep-modules.sh
# source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-amd-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-17ded13a78c26988d3e8ff72daa130124aa02ed8/scorep-modules.sh
module purge

rm -rf build CMakeCache.txt CMakeFiles
# source /ccs/proj/gen010/tools/install/config/config-scorep-9.0-rc1-PrgEnv-amd-8.6.0-cpe-24.11-rocm-6.3.1-papi-master-17ded13a78c26988d3e8ff72daa130124aa02ed8/scorep-modules.sh

module load craype-x86-trento \
            libfabric/1.22.0 \
            craype-network-ofi \
            Core/25.03 \
            tmux/3.4 \
            hsi/default \
            lfs-wrapper/0.0.1 \
            DefApps \
            cray-pmi/6.1.15 \
            perftools-base/24.11.0 \
            cpe/24.11 \
            craype/2.7.33 \
            cray-dsmml/0.3.0 \
            PrgEnv-amd/8.6.0 \
            amd/$ROCM_VERSION \
            cray-libsci/24.11.0 \
            cray-mpich-abi/8.1.31 \
            cray-mpich/8.1.31 \
            rocm/$ROCM_VERSION \
            cray-openshmemx/11.7.3 \

source ../../setup-env.sh

export OpenMP_CC=scorep-cc
export OpenMP_CXX=scorep-CC
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
#export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--libwrap=blis"
export CRAY_MPICH_PREFIX=$(dirname $(dirname $(which mpicc)))
export ROCHPL_ROOT=$(pwd)
export PREFIX=$ROCHPL_ROOT/install-scorep-amd
cd $ROCHPL_ROOT
rm -rf build
./install_scorep.sh \
   --with-rocm=${ROCM_PATH} \
   --with-rocblas=${ROCM_PATH} \
   --with-mpi=${CRAY_MPICH_PREFIX} \
   --prefix=$PREFIX

# cd $ROCHPL_ROOT/install-pat
# pat_build -g hip,io,mpi -w -f bin/rochpl

export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$ROCHPL_ROOT/tpl/blis/lib/:$LD_LIBRARY_PATH

#srun  -n 8  -c 8 --gpu-bind=closest $ROCHPL_ROOT/install/bin/rochpl -P 4 -Q 2 -p 4 -q 2 -N 128000 --NB 512
