cmake_minimum_required(VERSION 3.25)
project(ASE VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# =============================================================================
# Options
# =============================================================================
# The LAPACK implementation is selected with -DBLA_VENDOR=<...>, e.g.
# Intel10_64lp (MKL), OpenBLAS, Generic.
option(ASE_BUILD_TESTS   "Build the test binary"                   ON)
option(ASE_BUILD_BENCH   "Build the benchmark binary"              ON)
option(ASE_BUILD_DOCS    "Add the Doxygen `docs` target"           OFF)
option(ASE_WITH_OPENMP   "Use OpenMP for the D&C solver"           ON)
option(BUILD_SHARED_LIBS "Build ase as a shared library"           ON)

# =============================================================================
# Dependencies
# =============================================================================
find_package(CUDAToolkit REQUIRED)
find_package(LAPACK REQUIRED)

if(ASE_WITH_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "ASE_WITH_OPENMP=ON but OpenMP not found, thus we stay serial")
  endif()
endif()

# =============================================================================
# ase — symmetric eigensolver library
# =============================================================================
add_library(ase
  src/handle.cu
  src/cublas.cu
  src/cusolver.cu
  src/dbbr.cu
  src/bulge.cu
  src/secular.cu
  src/tridi.cu
  src/backtransform.cu
  src/solver.cu
)
add_library(ase::ase ALIAS ase)

target_include_directories(ase
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ase
  PUBLIC  CUDA::cublas CUDA::cusolver
  PRIVATE LAPACK::LAPACK)

set_target_properties(ase PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON)

if(ASE_WITH_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(ase PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(ase PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
endif()

# =============================================================================
# Benchmarks / tests
# =============================================================================
if(ASE_BUILD_BENCH)
  add_executable(ase_bench bench/bench.cpp)
  target_link_libraries(ase_bench PRIVATE ase)
endif()

if(ASE_BUILD_TESTS)
  enable_testing()
  add_executable(ase_test test/test.cu)
  target_link_libraries(ase_test PRIVATE ase)
  set_target_properties(ase_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME ase_test COMMAND ase_test)
endif()

# =============================================================================
# Doxygen
# =============================================================================
if(ASE_BUILD_DOCS)
  find_package(Doxygen QUIET OPTIONAL_COMPONENTS dot)
  if(DOXYGEN_FOUND)
    add_custom_target(docs
      COMMAND           ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT           "Running Doxygen -> ${CMAKE_SOURCE_DIR}/docs/html")
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/doxygen/doxygen-awesome.css)
      message(WARNING "doxygen/ submodule not checked out; docs will be unstyled. "
                      "Run: git submodule update --init")
    endif()
  else()
    message(WARNING "ASE_BUILD_DOCS=ON but Doxygen not found; no `docs` target")
  endif()
endif()

# =============================================================================
# Format
# =============================================================================
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ASE_FORMAT_SOURCES
    "${CMAKE_SOURCE_DIR}/include/ase/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.cu"      "${CMAKE_SOURCE_DIR}/src/*.cuh"
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/test/*.cu"     "${CMAKE_SOURCE_DIR}/test/*.h"
    "${CMAKE_SOURCE_DIR}/bench/*.cpp"   "${CMAKE_SOURCE_DIR}/bench/*.h")
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${ASE_FORMAT_SOURCES}
    COMMENT "clang-format -i")
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT} --dry-run --Werror ${ASE_FORMAT_SOURCES}
    COMMENT "clang-format --dry-run --Werror")
else()
  message(STATUS "clang-format not found; no `format` / `format-check` targets")
endif()

# =============================================================================
# Configure summary
# =============================================================================
message(STATUS "")
message(STATUS "ASE ${PROJECT_VERSION}")
message(STATUS "  build type        ${CMAKE_BUILD_TYPE}")
if(BUILD_SHARED_LIBS)
  message(STATUS "  library           shared")
else()
  message(STATUS "  library           static")
endif()
message(STATUS "  CUDA compiler     ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA toolkit      ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA architecture ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA flags        ${CMAKE_CUDA_FLAGS}")
message(STATUS "  LAPACK            ${LAPACK_LIBRARIES}")
message(STATUS "  OpenMP            ${OpenMP_CXX_FOUND}")
message(STATUS "  tests / bench     ${ASE_BUILD_TESTS} / ${ASE_BUILD_BENCH}")
message(STATUS "  docs              ${ASE_BUILD_DOCS}")
message(STATUS "")
