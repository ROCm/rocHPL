# Modifications (c) 2018-2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

option(HPL_VERBOSE_PRINT   "Enable printing to terminal during run" OFF)
option(HPL_PROGRESS_REPORT "Enable printing progess report to terminal during run" OFF)
option(HPL_DETAILED_TIMING "Enable detailed timers during run" OFF)
option(HPL_TRACING         "Annotate profiler traces with rocTX markers" OFF)

option(ROCM_PATH "Path to ROCm install" /opt/rocm)
option(HPL_MPI_DIR  "Path to MPI library" ${CMAKE_CURRENT_SOURCE_DIR}/tpl/openmpi)

set(CMAKE_INSTALL_PREFIX "rocHPL" CACHE PATH "Install path prefix, prepended onto install directories")

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${ROCM_PATH}/hip/cmake)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocHPL project
project(rochpl LANGUAGES CXX HIP)

# Build flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
include(cmake/Dependencies.cmake)

# Setup version
rocm_setup_version(VERSION 7.0.1)

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# HPL sources
file(GLOB_RECURSE rochpl_source RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")

# HPL device sources
file(GLOB_RECURSE rochpl_device_source RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*_device.cpp")
list(REMOVE_ITEM rochpl_source ${rochpl_device_source})

# Flag source files as hip source files
foreach(i ${rochpl_device_source})
  set_source_files_properties(${i} PROPERTIES LANGUAGE HIP)
endforeach()

list(APPEND ROCHPL_COMPILE_FLAGS -Wno-deprecated-declarations)
list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-fPIE>)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-ggdb>)
else()
  list(APPEND ROCHPL_COMPILE_FLAGS -march=native)
  list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=fast>)
  list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
  list(APPEND ROCHPL_COMPILE_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-funsafe-math-optimizations>)
endif()

# Target executable
add_executable(rochpl ${rochpl_source} ${rochpl_device_source})

target_compile_options(rochpl PRIVATE ${ROCHPL_COMPILE_FLAGS})

if(HPL_VERBOSE_PRINT)
  target_compile_definitions(rochpl PRIVATE HPL_VERBOSE_PRINT)
endif()

if(HPL_DETAILED_TIMING)
  target_compile_definitions(rochpl PRIVATE HPL_DETAILED_TIMING)
endif()

if(HPL_PROGRESS_REPORT)
  target_compile_definitions(rochpl PRIVATE HPL_PROGRESS_REPORT)
endif()

if(HPL_TRACING)
  target_compile_definitions(rochpl PRIVATE HPL_TRACING)
endif()

# Target include directories
target_include_directories(rochpl
                           PRIVATE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
if(HPL_TRACING)
  target_include_directories(rochpl
                             PRIVATE
                             $<BUILD_INTERFACE:${ROCM_PATH}/include/roctracer>)
endif()

#HIP
target_link_libraries(rochpl PRIVATE hip::host)

# MPI
target_link_libraries(rochpl PRIVATE MPI::MPI_CXX)

# OpenMP
target_link_libraries(rochpl PRIVATE OpenMP::OpenMP_CXX)

# Target link libraries
target_link_libraries(rochpl PRIVATE roc::rocblas)

if(HPL_TRACING)
  target_link_libraries(rochpl PRIVATE roc::roctracer)
  target_link_libraries(rochpl PRIVATE roc::roctx)
endif()

# Target properties
set_target_properties(rochpl PROPERTIES VERSION ${rochpl_VERSION})
set_target_properties(rochpl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set_target_properties(rochpl PROPERTIES LINKER_LANGUAGE CXX)

target_link_options(rochpl PRIVATE "-fopenmp")

if(MPI_GTL)
  find_library(GTL_LIB libmpi_gtl_hsa.so.0 "${MPI_GTL}")
  target_link_libraries(rochpl PRIVATE "${GTL_LIB}")
endif()

# Configure a header file to pass the rocHPL version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hpl_version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/hpl_version.hpp")

# Configure run scripts
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_rochpl.in"
               "${CMAKE_BINARY_DIR}/run_rochpl"
               @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/mpirun_rochpl.in"
               "${CMAKE_BINARY_DIR}/mpirun_rochpl"
               @ONLY)

#move input file
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/HPL.dat
     DESTINATION ${CMAKE_BINARY_DIR})

# Install targets
rocm_install_targets(TARGETS rochpl)

install(PROGRAMS ${CMAKE_BINARY_DIR}/run_rochpl ${CMAKE_BINARY_DIR}/mpirun_rochpl
        DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_BINARY_DIR}/HPL.dat
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-dev (>= 3.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-dev >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rochpl)

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute HPL application"
  MAINTAINER "Noel Chalmers")
