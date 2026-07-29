/**
 * @file   bench.cpp
 * @brief  Benchmark ase::symm_eig_solve vs cuSOLVER dsyevd / ssyevd.
 *
 * Usage: cuBench [--n N] [--warmup W] [--iters I]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "handle.h"
#include "ase.h"
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <type_traits>
#include <vector>

// =============================================================================
// Utilities
// =============================================================================

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s) {
        cudaEventRecord(start, s);
    }
    float end(cudaStream_t s) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

template <typename T> static void fill_symmetric(std::vector<T> &h, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            h[i * n + j] = h[j * n + i] = (T)(rand() % 200 - 100) / T(100);
}

// =============================================================================
// cuSOLVER reference
// =============================================================================

template <typename T>
static float bench_cusolver(cusolverDnHandle_t handle, int n, int warmup, int iters,
                            cudaStream_t stream) {
    std::vector<T> hA(n * n);
    fill_symmetric(hA, n);

    T *dA, *d_eval, *d_work;
    int *d_info;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_eval, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    int lwork = 0;
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, d_eval, &lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, d_eval, &lwork));
    }
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(T)));

    auto run = [&] {
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * n * sizeof(T), cudaMemcpyHostToDevice));
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_LOWER, n, dA, n, d_eval, d_work, lwork,
                                            d_info));
        } else {
            CUSOLVER_CHECK(cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_LOWER, n, dA, n, d_eval, d_work, lwork,
                                            d_info));
        }
    };

    for (int i = 0; i < warmup; ++i)
        run();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i)
        run();
    float ms = timer.end(stream);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    return ms / iters;
}

// =============================================================================
// ase
// =============================================================================

template <typename T> static float bench_ase(int n, int warmup, int iters, cudaStream_t stream) {
    std::vector<T> hA(n * n);
    fill_symmetric(hA, n);

    T *dA, *d_eval, *d_evec;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_eval, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_evec, n * n * sizeof(T)));

    auto run = [&] {
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * n * sizeof(T), cudaMemcpyHostToDevice));
        ase::symm_eig_solve<T>(dA, n, d_eval, d_evec, stream);
    };

    for (int i = 0; i < warmup; ++i)
        run();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i)
        run();
    float ms = timer.end(stream);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_evec));
    return ms / iters;
}

// =============================================================================
// main
// =============================================================================

template <typename T>
static void run_suite(cusolverDnHandle_t cusolver, int n, int warmup, int iters,
                      cudaStream_t stream) {
    const char *prec = std::is_same_v<T, float> ? "fp32" : "fp64";
    printf("=== solve %s  n=%d ===\n", prec, n);

    double flops = 4.0 / 3.0 * (double)n * n * n;

    float ms_ref = bench_cusolver<T>(cusolver, n, warmup, iters, stream);
    float ms_ase = bench_ase<T>(n, warmup, iters, stream);

    printf("  %-28s  %8.3f ms   %6.3f TFLOP/s\n",
           std::is_same_v<T, float> ? "cusolver_ssyevd" : "cusolver_dsyevd", ms_ref,
           flops / (ms_ref * 1e-3) / 1e12);
    printf("  %-28s  %8.3f ms   %6.3f TFLOP/s\n",
           std::is_same_v<T, float> ? "ase_symm_eig_solve<float>" : "ase_symm_eig_solve<double>",
           ms_ase, flops / (ms_ase * 1e-3) / 1e12);
    printf("\n");
}

int main(int argc, char **argv) {
    int n = 4096, warmup = 1, iters = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    }
    printf("cuBench  n=%d  warmup=%d  iters=%d\n\n", n, warmup, iters);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

    run_suite<float>(cusolver, n, warmup, iters, stream);
    run_suite<double>(cusolver, n, warmup, iters, stream);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
