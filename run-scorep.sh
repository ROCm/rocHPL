# source ./config-scorep.sh

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

source ../setup-run-params.sh

# Change directory to the script's location
cd "$(dirname "$0")"

export OpenMP_CC=scorep-cc
export OpenMP_CXX=scorep-CC
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
export CRAY_MPICH_PREFIX=$(dirname $(dirname $(which mpicc)))
export ROCHPL_ROOT=$(pwd)

export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$ROCHPL_ROOT/tpl/blis/lib/:$LD_LIBRARY_PATH

srun -A gen010 -t10 -N 2 --ntasks-per-node=4 --gpu-srange=800-801 --gpu-freq=800 --gpu-bind=closest $ROCHPL_ROOT/install-scorep-amd/bin/rochpl -P 2 -Q 4 -N 240000 --NB 512
