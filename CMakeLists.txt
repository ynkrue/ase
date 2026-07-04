cmake_minimum_required(VERSION 3.25)
project(cuEV LANGUAGES CXX CUDA)

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
option(CUEV_ENABLE_OMP    "Multithreaded BLAS/LAPACK"          OFF)
option(CUEV_ENABLE_TESTS  "Build tests"                         ON)
option(CUEV_ENABLE_BENCH  "Build benchmarks"                    ON)
option(CUEV_BUILD_DOCS    "Build Doxygen documentation"        OFF)

# =============================================================================
# Dependencies
# =============================================================================
find_package(CUDAToolkit REQUIRED)
# use multithreaded BLAS/LAPACK
add_library(cuev_lapack INTERFACE)
if(CUEV_ENABLE_OMP)
  set(MKL_INTERFACE lp64)
  set(MKL_THREADING gnu_thread)
  find_package(MKL CONFIG)
endif()
if(MKL_FOUND)
  message(STATUS "cuev_lapack: GNU-OpenMP threaded MKL")
  target_link_libraries(cuev_lapack INTERFACE MKL::MKL)
else()
  if(CUEV_ENABLE_OMP)
    message(WARNING "CUEV_ENABLE_OMP set but MKL not found; falling back to plain LAPACK")
  endif()
  find_package(LAPACK REQUIRED)
  target_link_libraries(cuev_lapack INTERFACE LAPACK::LAPACK)
endif()

# =============================================================================
# cuev — single-GPU library
# =============================================================================
add_library(cuev SHARED
  src/cuda/handle.cu
  src/cuda/cublas.cu
  src/cuda/cusolver.cu
  src/cuda/dbbr.cu
  src/cuda/bulge.cu
  src/cuda/tridi.cu
  src/cuda/backtransform.cu
  src/cuda/solver.cu
)
target_include_directories(cuev PUBLIC include)
target_link_libraries(cuev PUBLIC CUDA::cublas CUDA::cusolver cuev_lapack)
set_target_properties(cuev PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON)
# OpenMP for the host-side D&C merge loops (secular roots / eigenvector recon).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cuev PUBLIC OpenMP::OpenMP_CXX)
  target_compile_options(cuev PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

# =============================================================================
# Benchmarks
# =============================================================================
if(CUEV_ENABLE_BENCH)
  add_executable(cuBench bench/bench.cpp)
  target_link_libraries(cuBench PRIVATE cuev CUDA::cudart CUDA::cublas CUDA::cusolver)
  add_executable(cuBenchSolver bench/bench_solver.cpp)
  target_link_libraries(cuBenchSolver PRIVATE cuev CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()

# =============================================================================
# Tests
# =============================================================================
if(CUEV_ENABLE_TESTS)
  add_executable(cuTest
    test/cuda/test_dbbr.cu
    test/cuda/test_bulge.cu
    test/cuda/test_backtransform.cu
    test/cuda/test_tridi.cu
  )
  target_link_libraries(cuTest PRIVATE cuev CUDA::cudart CUDA::cublas CUDA::cusolver)
  set_target_properties(cuTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# =============================================================================
# Doxygen
# =============================================================================
if(CUEV_BUILD_DOCS)
  find_package(Doxygen QUIET OPTIONAL_COMPONENTS dot)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_PROJECT_NAME        "cuEV")
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
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/bench/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
    COMMENT "Running clang-format")
endif()
