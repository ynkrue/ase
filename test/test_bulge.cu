/**
 * @file   test_bulge.cu
 * @brief  Correctness tests for bulge chasing (bc_pack + bc_chase): band → tridiagonal.
 *
 * Validates by spectrum preservation: eigenvalues of the (d,e) tridiagonal produced
 * by BC must match cuSOLVER's eigenvalues of the original symmetric matrix.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cusolverDn.h>
#include <type_traits>
#include <vector>

using namespace cutest;

// Ascending eigenvalues of a symmetric matrix (lower triangle) via cuSOLVER syevd.
template <typename T> static std::vector<T> eig_cusolver(const std::vector<T> &hA, int n) {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);
    T *dA = to_device(hA), *dW, *dwork;
    int *dInfo, lwork = 0;
    cudaMalloc(&dW, n * sizeof(T));
    cudaMalloc(&dInfo, sizeof(int));
    auto job = CUSOLVER_EIG_MODE_NOVECTOR;
    auto uplo = CUBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float>)
        cusolverDnSsyevd_bufferSize(h, job, uplo, n, dA, n, dW, &lwork);
    else
        cusolverDnDsyevd_bufferSize(h, job, uplo, n, dA, n, dW, &lwork);
    cudaMalloc(&dwork, lwork * sizeof(T));
    if constexpr (std::is_same_v<T, float>)
        cusolverDnSsyevd(h, job, uplo, n, dA, n, dW, dwork, lwork, dInfo);
    else
        cusolverDnDsyevd(h, job, uplo, n, dA, n, dW, dwork, lwork, dInfo);
    std::vector<T> w(n);
    to_host(w, dW);
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dwork);
    cudaFree(dInfo);
    cusolverDnDestroy(h);
    return w; // ascending order
}

// Full BC: random symmetric A → DBBR band → bc_pack → bc_chase → (d,e).
// Build the tridiagonal from (d,e) and check its spectrum matches A's.
template <typename T> static void bulge_case(int n, int nbw, int nk, double tol) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = ase::handle_alloc<T>(n, nbw, nk, stream);

    std::vector<T> A0(n * n);
    fill_random(A0, 7);
    auto ev_ref = eig_cusolver(A0, n);

    T *dA = to_device(A0);
    ase::kernels::dbbr_reduce(&ws, dA, ws.B);
    ase::kernels::bc_chase(&ws, ws.B, ws.d, ws.e);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> d(n), e(n);
    to_host(d, ws.d);
    to_host(e, ws.e);

    // assemble full tridiagonal and take its spectrum
    std::vector<T> Tri(n * n, T(0));
    for (int i = 0; i < n; ++i)
        Tri[i + i * n] = d[i];
    for (int i = 0; i < n - 1; ++i) {
        Tri[(i + 1) + i * n] = e[i];
        Tri[i + (i + 1) * n] = e[i];
    }
    auto ev = eig_cusolver(Tri, n);

    double range = (double)ev_ref[n - 1] - (double)ev_ref[0];
    double maxdiff = 0;
    for (int i = 0; i < n; ++i)
        maxdiff = std::max(maxdiff, std::abs((double)ev[i] - (double)ev_ref[i]));
    CHECK_LT(maxdiff / range, tol);

    CUDA_CHECK(cudaFree(dA));
    ase::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(bc_chase, fp64_oneblock) {
    bulge_case<double>(256, 64, 256, 1e-9);
}
TEST(bc_chase, fp64_multiblock) {
    bulge_case<double>(512, 64, 256, 1e-9);
}
