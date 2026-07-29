cmake_minimum_required(VERSION 3.25)
project(ASE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# =============================================================================
# Options
# =============================================================================
option(ASE_ENABLE_OMP    "Multithreaded BLAS/LAPACK"          OFF)
option(ASE_ENABLE_TESTS  "Build tests"                         ON)
option(ASE_ENABLE_BENCH  "Build benchmarks"                    ON)
option(ASE_BUILD_DOCS    "Build Doxygen documentation"        OFF)

# =============================================================================
# Dependencies
# =============================================================================
find_package(CUDAToolkit REQUIRED)
# use multithreaded BLAS/LAPACK
add_library(ase_lapack INTERFACE)
if(ASE_ENABLE_OMP)
  set(MKL_INTERFACE lp64)
  set(MKL_THREADING gnu_thread)
  find_package(MKL CONFIG)
endif()
if(MKL_FOUND)
  message(STATUS "ase_lapack: GNU-OpenMP threaded MKL")
  target_link_libraries(ase_lapack INTERFACE MKL::MKL)
else()
  if(ASE_ENABLE_OMP)
    message(WARNING "ASE_ENABLE_OMP set but MKL not found; falling back to plain LAPACK")
  endif()
  find_package(LAPACK REQUIRED)
  target_link_libraries(ase_lapack INTERFACE LAPACK::LAPACK)
endif()

# =============================================================================
# ase — single-GPU symmetric eigensolver library
# =============================================================================
add_library(ase SHARED
  src/handle.cu
  src/cublas.cu
  src/cusolver.cu
  src/dbbr.cu
  src/bulge.cu
  src/tridi.cu
  src/backtransform.cu
  src/solver.cu
)
target_include_directories(ase PUBLIC include)
target_link_libraries(ase PUBLIC CUDA::cublas CUDA::cusolver ase_lapack)
set_target_properties(ase PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON)
# OpenMP for the host-side D&C merge loops (secular roots / eigenvector recon).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ase PUBLIC OpenMP::OpenMP_CXX)
  target_compile_options(ase PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(ASE_ENABLE_BENCH)
  add_executable(ase_bench bench/bench.cpp)
  target_link_libraries(ase_bench PRIVATE ase CUDA::cudart CUDA::cublas CUDA::cusolver)
  add_executable(ase_bench_solver bench/bench_solver.cpp)
  target_link_libraries(ase_bench_solver PRIVATE ase CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# =============================================================================
# Tests
# =============================================================================
if(ASE_ENABLE_TESTS)
  add_executable(ase_test
    test/test_dbbr.cu
    test/test_bulge.cu
    test/test_backtransform.cu
    test/test_tridi.cu
  )
  target_link_libraries(ase_test PRIVATE ase CUDA::cudart CUDA::cublas CUDA::cusolver)
  set_target_properties(ase_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# =============================================================================
# Doxygen
# =============================================================================
if(ASE_BUILD_DOCS)
  find_package(Doxygen QUIET OPTIONAL_COMPONENTS dot)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME        "ASE")
    set(DOXYGEN_PROJECT_BRIEF       "CUDA symmetric dense eigensolver")
    set(DOXYGEN_EXTRACT_ALL         YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_OUTPUT_DIRECTORY    "${CMAKE_BINARY_DIR}/docs")
    doxygen_add_docs(docs include src)
  endif()
endif()

# =============================================================================
# Format
# =============================================================================
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE FORMAT_SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/bench/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cu"
    "${CMAKE_SOURCE_DIR}/test/*.h"
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
    COMMENT "Running clang-format")
endif()
