#!/bin/bash
cd /scratch/yrfenach/applications/ase

export CUDA_HOME="$KEZ_WORKDIR/env/vendors/cuda-13.3"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export MKLROOT="$KEZ_WORKDIR/env/vendors/intel-oneapi-2026.1.0.192/mkl/latest"

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DBLA_VENDOR=Intel10_64lp_seq

cmake --build build -j16
