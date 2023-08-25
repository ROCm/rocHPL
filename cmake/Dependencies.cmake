# Modifications (c) 2019-2022 Advanced Micro Devices, Inc.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Dependencies

# Git
find_package(Git REQUIRED)

#Look for a BLAS lib
# For some reason cmake doesn't let us manually specify a search path in FindBLAS,
# so let's add our own libraries
get_filename_component(HPL_BLAS_DIR ${HPL_BLAS_DIR} ABSOLUTE)

# Look for BLIS in the provided path
find_library(BLAS_LIBRARIES NAMES blis
             PATHS ${HPL_BLAS_DIR}
             NO_DEFAULT_PATH)

if (NOT BLAS_LIBRARIES)
  # If we dont find BLIS, look for openblas
  find_library(BLAS_LIBRARIES NAMES openblas
               PATHS ${HPL_BLAS_DIR}
               NO_DEFAULT_PATH)
endif()
if (NOT BLAS_LIBRARIES)
  # If we dont find BLIS or openBLAS, look for MKL
  find_library(BLAS_LIBRARIES NAMES mkl_core
               PATHS ${HPL_BLAS_DIR}
               NO_DEFAULT_PATH)
  find_library(BLAS_SEQ_LIBRARIES NAMES mkl_sequential
               PATHS ${HPL_BLAS_DIR}
               NO_DEFAULT_PATH)
  find_library(BLAS_LP64_LIBRARIES NAMES mkl_intel_lp64
               PATHS ${HPL_BLAS_DIR}
               NO_DEFAULT_PATH)
endif()

if (BLAS_LIBRARIES)
  message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
else()
  # If we still havent found a blas library, maybe cmake will?
  find_package(BLAS REQUIRED)
endif()
add_library(BLAS::BLAS IMPORTED INTERFACE)
set_property(TARGET BLAS::BLAS PROPERTY INTERFACE_LINK_LIBRARIES  "${BLAS_LP64_LIBRARIES};${BLAS_SEQ_LIBRARIES};${BLAS_LIBRARIES}")

# Find OpenMP package
find_package(OpenMP)
if (NOT OPENMP_FOUND)
  message("-- OpenMP not found. Compiling WITHOUT OpenMP support.")
else()
  option(HPL_OPENMP "Compile WITH OpenMP support." ON)
endif()

# MPI
set(MPI_HOME ${HPL_MPI_DIR})
find_package(MPI REQUIRED)

# Add some paths
list(APPEND CMAKE_PREFIX_PATH ${ROCBLAS_PATH} ${ROCM_PATH} )
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip )

find_library(ROCTRACER NAMES roctracer64
             PATHS ${ROCM_PATH}/lib
             NO_DEFAULT_PATH)
find_library(ROCTX NAMES roctx64
             PATHS ${ROCM_PATH}/lib
             NO_DEFAULT_PATH)

message("-- roctracer:  ${ROCTRACER}")
message("-- roctx:      ${ROCTX}")

add_library(roc::roctracer SHARED IMPORTED)
set_target_properties(roc::roctracer PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ROCM_PATH}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
  IMPORTED_LOCATION "${ROCTRACER}"
  IMPORTED_SONAME "libroctracer.so")
add_library(roc::roctx SHARED IMPORTED)
set_target_properties(roc::roctx PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ROCM_PATH}/include"
  INTERFACE_LINK_LIBRARIES "hip::host"
  IMPORTED_LOCATION "${ROCTX}"
  IMPORTED_SONAME "libroctx64.so")

# Find HIP package
find_package(HIP REQUIRED)

# rocblas
find_package(rocblas REQUIRED)

get_target_property(rocblas_LIBRARIES roc::rocblas IMPORTED_LOCATION_RELEASE)

message("-- rocBLAS version:      ${rocblas_VERSION}")
message("-- rocBLAS include dirs: ${rocblas_INCLUDE_DIRS}")
message("-- rocBLAS libraries:    ${rocblas_LIBRARIES}")

get_filename_component(ROCBLAS_LIB_PATH ${rocblas_LIBRARIES} DIRECTORY)

# ROCm cmake package
find_package(ROCM QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCM_FOUND)
  set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package(ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds OPTIONAL)
