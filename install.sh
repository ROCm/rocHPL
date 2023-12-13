#!/usr/bin/env bash
# Author: Nico Trost
# Modified by: Noel Chalmers

#set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocHPL build helper script"
  echo "./install "
  echo "    [-h|--help] prints this help message"
  echo "    [-g|--debug] Set build type to Debug (otherwise build Release)"
  echo "    [--prefix] Path to rocHPL install location (Default: build/rocHPL)"
  echo "    [--with-rocm=<dir>] Path to ROCm install (Default: /opt/rocm)"
  echo "    [--with-rocblas=<dir>] Path to rocBLAS library (Default: /opt/rocm/rocblas)"
  echo "    [--with-cpublas=<dir>] Path to external CPU BLAS library (Default: clone+build BLIS)"
  echo "    [--with-mpi=<dir>] Path to external MPI install (Default: clone+build OpenMPI)"
  echo "    [--verbose-print] Verbose output during HPL setup (Default: true)"
  echo "    [--progress-report] Print progress report to terminal during HPL run (Default: true)"
  echo "    [--detailed-timing] Record detailed timers during HPL run (Default: true)"
}

# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora and SLES\n"
        exit 2
        ;;
  esac
}

exit_with_error( )
{
  if (( $1 == 2 )); then
    # Failure in some install step
    # Print some message about needed dependencies

    # dependencies needed for executable to build
    local library_dependencies_ubuntu=( "git" "make" "cmake" "libnuma-dev" "pkg-config" "autoconf" "libtool" "automake" "m4" "flex" "libgomp1")
    local library_dependencies_centos=( "git" "make" "cmake3" "gcc-c++" "rpm-build" "epel-release" "numactl-libs" "autoconf" "libtool" "automake" "m4" "flex" "libgomp")
    local library_dependencies_fedora=( "git" "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" "numactl-libs"  "autoconf" "libtool" "automake" "m4" "flex" "libgomp")
    local library_dependencies_sles=(   "git" "make" "cmake" "gcc-c++" "libcxxtools9" "rpm-build" "libnuma-devel" "autoconf" "libtool" "automake" "m4" "flex" "libgomp1")

    if [[ "${with_rocm}" == /opt/rocm ]]; then
      library_dependencies_ubuntu+=("rocblas" "rocblas-dev")
      library_dependencies_centos+=("rocblas" "rocblas-devel")
      library_dependencies_fedora+=("rocblas" "rocblas-dev")
      library_dependencies_sles+=("rocblas" "rocblas-devel")
    fi

    printf "Installation failed. Some required packages may be missing.\n"
    printf "The following package manager install command may be needed:\n"
    case "${ID}" in
      ubuntu)
        printf "sudo apt install -y ${library_dependencies_ubuntu[*]}\n"
        ;;

      centos|rhel)
        printf "sudo yum -y --nogpgcheck install ${library_dependencies_centos[*]}\n"
        ;;

      fedora)
        printf "sudo dnf install -y ${library_dependencies_fedora[*]}\n"
        ;;

      sles)
        printf "sudo zypper -n --no-gpg-checks install ${library_dependencies_sles[*]}\n"
        ;;
      *)
        exit 2
        ;;
    esac
  fi

  exit $1
}

check_exit_code( )
{
  if (( $? != 0 )); then
    err=$1
    msg=$2
    if [[ "$msg" == "" ]]; then
      msg="Unknown error"
    fi
    echo "ERROR: $msg"
    exit $err
  fi
}


# Install BLIS in rochpl/tpl
install_blis( )
{
  if [ ! -d "./tpl/blis" ]; then
    mkdir -p tpl && cd tpl
    git clone https://github.com/amd/blis --branch 4.1
    check_exit_code 2
    cd blis; ./configure --prefix=${PWD} --enable-cblas --disable-sup-handling auto;
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install -j$(nproc)
    check_exit_code 2
    cd ../..
  elif [ ! -f "./tpl/blis/lib/libblis.so" ]; then
    cd tpl/blis; ./configure --prefix=${PWD} --enable-cblas --disable-sup-handling auto;
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install -j$(nproc)
    check_exit_code 2
    cd ../..
  fi

  # Check for successful build
  if [ ! -f "./tpl/blis/lib/libblis.so" ]; then
    echo "Error: BLIS install unsuccessful."
    exit_with_error 2
  fi
}

# Clone and build OpenMPI+UCX in rochpl/tpl
install_openmpi( )
{
  #OpenMPI and UCX install to one of these locations depending on OS
  ucx_lib_folder=./tpl/ucx/lib
  ompi_lib_folder=./tpl/openmpi/lib
  ucx_lib64_folder=./tpl/ucx/lib64
  ompi_lib64_folder=./tpl/openmpi/lib64

  if [ ! -d "./tpl/ucx" ]; then
    mkdir -p tpl && cd tpl
    git clone --branch v1.14.1 https://github.com/openucx/ucx.git ucx
    check_exit_code 2
    cd ucx;
    ./autogen.sh; ./autogen.sh #why do we have to run this twice?
    check_exit_code 2
    mkdir build; cd build
    ../contrib/configure-opt --prefix=${PWD}/../ --with-rocm=${with_rocm} --without-knem --without-cuda --without-java
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install
    check_exit_code 2
    cd ../../..
  elif ([ ! -f "${ucx_lib_folder}/libucm.so" ] || [ ! -f "${ucx_lib_folder}/libucp.so" ]  || \
        [ ! -f "${ucx_lib_folder}/libucs.so" ] || [ ! -f "${ucx_lib_folder}/libuct.so" ]) && \
       ([ ! -f "${ucx_lib64_folder}/libucm.so" ] || [ ! -f "${ucx_lib64_folder}/libucp.so" ]  || \
        [ ! -f "${ucx_lib64_folder}/libucs.so" ] || [ ! -f "${ucx_lib64_folder}/libuct.so" ]); then
    cd tpl/ucx; 
    ./autogen.sh; ./autogen.sh
    check_exit_code 2
    mkdir build; cd build
    ../contrib/configure-opt --prefix=${PWD}/../ --with-rocm=${with_rocm} --without-knem --without-cuda --without-java
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install
    check_exit_code 2
    cd ../../..
  fi

  # Check for successful build
  if ([ ! -f "${ucx_lib_folder}/libucm.so" ] || [ ! -f "${ucx_lib_folder}/libucp.so" ]  || \
      [ ! -f "${ucx_lib_folder}/libucs.so" ] || [ ! -f "${ucx_lib_folder}/libuct.so" ]) &&
     ([ ! -f "${ucx_lib64_folder}/libucm.so" ] || [ ! -f "${ucx_lib64_folder}/libucp.so" ]  || \
      [ ! -f "${ucx_lib64_folder}/libucs.so" ] || [ ! -f "${ucx_lib64_folder}/libuct.so" ]); then
    echo "Error: UCX install unsuccessful."
    exit 3
  fi

  if [ ! -d "./tpl/openmpi" ]; then
    mkdir -p tpl && cd tpl
    git clone --branch v4.1.5 https://github.com/open-mpi/ompi.git openmpi
    check_exit_code 2
    cd openmpi; ./autogen.pl;
    check_exit_code 2
    mkdir build; cd build
    ../configure --prefix=${PWD}/../ --with-ucx=${PWD}/../../ucx --without-verbs
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install
    check_exit_code 2
    cd ../../..
  elif [ ! -f "${ompi_lib_folder}/libmpi.so" ] && [ ! -f "${ompi_lib64_folder}/libmpi.so" ]; then
    cd tpl/openmpi; ./autogen.pl;
    check_exit_code 2
    mkdir build; cd build
    ../configure --prefix=${PWD}/../ --with-ucx=${PWD}/../../ucx --without-verbs
    check_exit_code 2
    make -j$(nproc)
    check_exit_code 2
    make install
    check_exit_code 2
    cd ../../..
  fi

  # Check for successful build
  if [ ! -f "${ompi_lib_folder}/libmpi.so" ] && [ ! -f "${ompi_lib64_folder}/libmpi.so" ]; then
    echo "Error: OpenMPI install unsuccessful."
    exit_with_error 2
  fi
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit_with_error 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit_with_error 1
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_prefix=rocHPL
build_release=true
with_rocm=/opt/rocm
with_mpi=tpl/openmpi
with_rocblas=/opt/rocm/rocblas
with_cpublas=tpl/blis/lib
verbose_print=true
progress_report=true
detailed_timing=true

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,debug,prefix:,with-rocm:,with-mpi:,with-rocblas:,with-cpublas:,verbose-print:,progress-report:,detailed-timing: --options hg -- "$@")
else
  echo "Need a new version of getopt"
  exit_with_error 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit_with_error 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -g|--debug)
        build_release=false
        shift ;;
    --prefix)
        install_prefix=${2}
        shift 2 ;;
    --with-rocm)
        with_rocm=${2}
        shift 2 ;;
    --with-mpi)
        with_mpi=${2}
        shift 2 ;;
    --with-rocblas)
        with_rocblas=${2}
        shift 2 ;;
    --with-cpublas)
        with_cpublas=${2}
        shift 2 ;;
    --verbose-print)
        verbose_print=${2}
        shift 2 ;;
    --progress-report)
        progress_report=${2}
        shift 2 ;;
    --detailed-timing)
        detailed_timing=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit_with_error 1
        ;;
  esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
rm -rf ${build_dir}

# Default cmake executable is called cmake
cmake_executable=cmake

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
export ROCM_PATH=${with_rocm}
export PATH=${PATH}:${ROCM_PATH}/bin

pushd .
  # #################################################
  # BLAS
  # #################################################
  if [[ "${with_cpublas}" == tpl/blis/lib ]]; then

    install_blis

  fi

  # #################################################
  # MPI
  # #################################################
  if [[ "${with_mpi}" == tpl/openmpi ]]; then

    with_mpi=${PWD}/tpl/openmpi
    install_openmpi

  fi

  # #################################################
  # configure & build
  # #################################################
  cmake_common_options="-DCMAKE_INSTALL_PREFIX=${install_prefix} -DHPL_BLAS_DIR=${with_cpublas}
                        -DHPL_MPI_DIR=${with_mpi} -DROCM_PATH=${with_rocm} -DROCBLAS_PATH=${with_rocblas}"

  # build type
  if [[ "${build_release}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  shopt -s nocasematch
  if [[ "${verbose_print}" == on || "${verbose_print}" == true || "${verbose_print}" == 1 || "${verbose_print}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_VERBOSE_PRINT=ON"
  fi
  if [[ "${progress_report}" == on || "${progress_report}" == true || "${progress_report}" == 1 || "${progress_report}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_PROGRESS_REPORT=ON"
  fi
  if [[ "${detailed_timing}" == on || "${detailed_timing}" == true || "${detailed_timing}" == 1 || "${detailed_timing}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_DETAILED_TIMING=ON"
  fi
  shopt -u nocasematch

  # Build library with AMD toolchain because of existence of device kernels
  mkdir -p ${build_dir} && cd ${build_dir}
  ${cmake_executable} ${cmake_common_options} ..
  check_exit_code 2

  if [[ -e build.ninja ]]; then
    command -v ninja > /dev/null 2>&1
    check_exit_code 2 "Ninja command was not found, but is required by CMake config"
    ninja install
  else
    make -j$(nproc) install
  fi
  check_exit_code 2

popd
