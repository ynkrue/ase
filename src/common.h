#pragma once
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// =============================================================================
// Error checking
// =============================================================================

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t _e = (err);                                                                    \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t _e = (err);                                                                 \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                                         \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);              \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t _e = (err);                                                               \
        if (_e != CUSOLVER_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// =============================================================================
// Utilities
// =============================================================================

inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Device helpers  (all ASE translation units are .cu, so one __CUDACC__ block covers
// both the host-callable helpers below and the device-only ones)
// =============================================================================
#ifdef __CUDACC__

/// LAPACK's DLAMCH('Epsilon'): the relative machine precision, ulp/2 under IEEE
/// round-to-nearest. This is *half* of C's DBL_EPSILON — DBL_EPSILON is LAPACK's
/// DLAMCH('Precision'), a different quantity.
///
/// Every tolerance in the ported D&C code is a multiple of this: the deflation test in
/// tridi.cu's host_laed2, the block-splitting test in stedx, and the convergence tests
/// in secular.cu's dlaed4/dlaed6. Reaching for std::numeric_limits<double>::epsilon()
/// instead silently doubles all of them, so route every one of them through here.
__host__ __device__ __forceinline__ double lapack_eps() {
    return DBL_EPSILON / 2.0;
}

__device__ __forceinline__ double tabs(double x) {
    return x < 0.0 ? -x : x;
}

/// Reference to packed lower-band A[i,j] (i >= j): packed row = i-j, col = j, leading dim ldb.
__device__ __forceinline__ double &band_at(double *B, int i, int j, int ldb) {
    return B[(i - j) + j * ldb];
}

/// Symmetric read of A[i,j] (any i,j in band); reflects the upper triangle to the stored lower
/// band.
__device__ __forceinline__ double band_sym(const double *B, int i, int j, int ldb) {
    if (i < j) {
        int t = i;
        i = j;
        j = t;
    }
    return B[(i - j) + j * ldb];
}

/// Sum a value across the 32 lanes of a warp; every lane returns the total.
__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/// Block-wide sum reduction into thread 0. Caller broadcasts and syncs after.
template <int BLOCKSIZE>
__device__ __forceinline__ double block_reduce_sum(double val, double *smem) {
    smem[threadIdx.x] = val;
    __syncthreads();
    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    double v = 0.0;
    if (threadIdx.x < 32) {
        v = smem[threadIdx.x];
        v += __shfl_down_sync(0xffffffff, v, 16);
        v += __shfl_down_sync(0xffffffff, v, 8);
        v += __shfl_down_sync(0xffffffff, v, 4);
        v += __shfl_down_sync(0xffffffff, v, 2);
        v += __shfl_down_sync(0xffffffff, v, 1);
    }
    return v;
}

#endif // __CUDACC__
