# rocHPL
rocHPL is a benchmark based on the [HPL][] benchmark application, implemented on top of AMD's Radeon Open Compute [ROCm][] Platform, runtime, and toolchains. rocHPL is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.10 or later)
* MPI (Optional)
* AMD [ROCm] platform (3.5 or later)
* [rocBLAS][]

## Quickstart rocHPL build and install

#### Install script
You can build rocHPL using the `install.sh` script
```
# Clone rocHPL using git
git clone https://github.com/ROCm/rocHPL.git

# Go to rocHPL directory
cd rocHPL

# Run install.sh script
# Command line options:
#    -h|--help            - prints this help message
#    -g|--debug           - Set build type to Debug (otherwise build Release)
#    --prefix=<dir>       - Path to rocHPL install location (Default: build/rocHPL)
#    --with-rocm=<dir>    - Path to ROCm install (Default: /opt/rocm)
#    --with-rocblas=<dir> - Path to rocBLAS library (Default: /opt/rocm/rocblas)
#    --with-mpi=<dir>     - Path to external MPI install (Default: clone+build OpenMPI)
#    --verbose-print      - Verbose output during HPL setup (Default: true)
#    --progress-report    - Print progress report to terminal during HPL run (Default: true)
#    --detailed-timing    - Record detailed timers during HPL run (Default: true)
#    --enable-tracing     - Annotate profiler traces with rocTX markers (Default: false)
./install.sh
```
By default, [UCX] v1.16.0, and [OpenMPI] v5.0.3 will be cloned and built in rocHPL/tpl. After building, the `rochpl` executable is placed in build/rochpl-install.

## Running rocHPL benchmark application
rocHPL provides some helpful wrapper scripts. A wrapper script for launching via `mpirun` is provided in `mpirun_rochpl`. This script has two distinct run modes:
```
mpirun_rochpl -P <P> -Q <P> -N <N> --NB <NB> -f <frac>
# where
# P       - is the number of rows in the MPI grid
# Q       - is the number of columns in the MPI grid
# N       - is the total number of rows/columns of the global matrix
# NB      - is the panel size in the blocking algorithm
# frac    - is the split-update fraction (imporant for hiding some MPI
            communication)
```
This run script will launch a total of np=PxQ MPI processes.

The second runmode takes an input file together with a number of MPI processes:
```
mpirun_rochpl -P <p> -Q <q> -i <input> -f <frac>
# where
# P       - is the number of rows in the MPI grid
# Q       - is the number of columns in the MPI grid
# input   - is the input filename (default HPL.dat)
# frac    - is the split-update fraction (important for hiding some MPI
            communication)
```

The input file accpted by the `rochpl` executable follows the format below:
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
0            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
45312        Ns
1            # of NBs
384          NBs
1            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
32           NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
2            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
1            DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

The `mpirun_rochpl` wraps a second script, `run_rochpl`, wherein some CPU core bindings are determined autmotically based on the node-local MPI grid. 

Users wishing to launch rocHPL via a workload manager such as slurm may launch the `run_rochpl` script, or may launch the `rochpl` binary directly and specify CPU+GPU bindings via the job manager. For example:
```
srun -N 2 -n 16 -c 16 --gpus-per-task 1 --gpu-bind=closest ./build/bin/rochpl -P 4 -Q 4 -N 128000 -NB 512
```
When launching to multiple compute nodes, it can be useful to specify the local MPI grid layout on each node. To specify this, the `-p` and `-q` input parameters are used. For example, the srun line above is launching to two compute nodes, each with 8 GPUs. The local MPI grid layout can be specifed as either:
```
srun -N 2 -n 16 -c 16 --gpus-per-task 1 --gpu-bind=closest ./build/bin/rochpl -P 4 -Q 4 -p 2 -q 4 -N 128000 -NB 512
```
or
```
srun -N 2 -n 16 -c 16 --gpus-per-task 1 --gpu-bind=closest ./build/bin/rochpl -P 4 -Q 4 -p 4 -q 2 -N 128000 -NB 512
```
This helps to control where/how much inter-node communication is occuring.

## Performance evaluation
rocHPL is typically weak scaled so that the global matrix fills all available VRAM on all GPUs. The matrix size N is usually selected to be a multiple of the blocksize NB. Some sample runs on 32GB MI100 GPUs include:
* 1 MI100: `mpirun_rochpl -P 1 -Q 1 -N  64512 --NB 512`
* 2 MI100: `mpirun_rochpl -P 1 -Q 2 -N  90112 --NB 512`
* 4 MI100: `mpirun_rochpl -P 2 -Q 2 -N 126976 --NB 512`
* 8 MI100: `mpirun_rochpl -P 2 -Q 4 -N 180224 --NB 512`

Overall performance of the benchmark is measured in 64-bit floating point operations (FLOPs) per second. Performance is reported at the end of the run to the user's specified output (by default the performance is printed to stdout and a results file HPL.out).

See [the Wiki](../../wiki/Common-rocHPL-run-configurations) for some common run configurations for various AMD Instinct GPUs.

## Testing rocHPL
At the end of each benchmark run, residual error checking is computed, and PASS or FAIL is printed to output.

The simplest suite of tests should run configurations from 1 to 4 GPUs to exercise different communcation code paths. For example the tests:
```
mpirun_rochpl -P 1 -Q 1 -N 45312
mpirun_rochpl -P 1 -Q 2 -N 45312
mpirun_rochpl -P 2 -Q 1 -N 45312
mpirun_rochpl -P 2 -Q 2 -N 45312
```
should all report PASSED.

Please note that for successful testing, a device with at least 16GB of device memory is required.

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[HPL]: http://icl.utk.edu/hpl/
[ROCm]: https://github.com/ROCm/ROCm
[HIP]: https://github.com/ROCm/HIP
[rocBLAS]: https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas
[OpenMPI]: https://github.com/open-mpi/ompi
[UCX]: https://github.com/openucx/ucx
[the issue tracker]: https://github.com/ROCm/rocHPL/issues
[license file]: https://github.com/ROCm/rocHPL
