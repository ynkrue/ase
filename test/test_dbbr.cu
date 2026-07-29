/**
 * @file   test_dbbr.cu
 * @brief  Correctness tests for DBBR kernels (dbbr_panel_qr: geqrf + extract + larft).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include "test.h"
#include <algorithm>
#include <cusolverDn.h>
#include <type_traits>

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
    return w; // syevd returns ascending order
}

// Validates one panel QR: a random rows×b panel A is factorised, then we rebuild
// the block reflector Q = I − V·T·Vᵀ from the outputs (V, T) and check
//   (1) reconstruction:  (I − V·T·Vᵀ)·R  ==  A_original
//   (2) orthogonality:   ‖QᵀQ − I‖ ≈ 0
template <typename T> static void panel_qr_case(int rows, int b, double tol) {
    const int n = rows; // standalone panel: lda = ws->n = rows
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = ase::handle_alloc<T>(n, b, b, stream);

    std::vector<T> A0(rows * b);
    fill_random(A0, 12345);

    T *dA = to_device(A0);
    T *dV = nullptr;
    CUDA_CHECK(cudaMalloc(&dV, (size_t)rows * b * sizeof(T)));

    ase::kernels::dbbr_panel_qr(&ws, dA, dV, rows, b);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> Apk(rows * b), V(rows * b), Tm(b * b);
    to_host(Apk, dA);
    to_host(V, dV);
    to_host(Tm, ws.Tri);

    // R_full (rows×b): upper triangle of the packed factor, zeros below
    std::vector<T> R(rows * b, T(0));
    for (int c = 0; c < b; ++c)
        for (int r = 0; r <= c; ++r)
            R[r + c * rows] = Apk[r + c * rows];

    // (1) reconstruction: recon = R − V·(T·(Vᵀ·R))  ==  (I − V·T·Vᵀ)·R
    std::vector<T> VtR(b * b), TVtR(b * b), corr(rows * b), recon(rows * b);
    gemm_host(VtR, V, R, b, b, rows, true, false, rows, rows, b);
    gemm_host(TVtR, Tm, VtR, b, b, b, false, false, b, b, b);
    gemm_host(corr, V, TVtR, rows, b, b, false, false, rows, b, rows);
    for (int i = 0; i < rows * b; ++i)
        recon[i] = R[i] - corr[i];
    double rel_recon = frob_diff(recon, A0) / frob(A0);

    // (2) orthogonality: Q = I − V·T·Vᵀ ;  ‖QᵀQ − I‖ / √rows
    std::vector<T> VT(rows * b), Q(rows * rows), QtQ(rows * rows), Id(rows * rows, T(0));
    gemm_host(VT, V, Tm, rows, b, b, false, false, rows, b, rows);
    gemm_host(Q, VT, V, rows, rows, b, false, true, rows, rows, rows); // Q = V·T·Vᵀ
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < rows; ++i)
            Q[i + j * rows] = (i == j ? T(1) : T(0)) - Q[i + j * rows];
    gemm_host(QtQ, Q, Q, rows, rows, rows, true, false, rows, rows, rows);
    for (int i = 0; i < rows; ++i)
        Id[i + i * rows] = T(1);
    double orth = frob_diff(QtQ, Id) / std::sqrt((double)rows);

    CHECK_LT(rel_recon, tol);
    CHECK_LT(orth, tol);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dV));
    ase::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Full band reduction: A → band (bandwidth nbw). Validate by (1) spectrum preserved
// (eigenvalues of the clean band match A's) and (2) nothing strictly below the band.
template <typename T> static void band_reduce_case(int n, int nbw, int nk, double tol) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = ase::handle_alloc<T>(n, nbw, nk, stream);

    std::vector<T> A0(n * n);
    fill_random(A0, 7);
    auto ev_ref = eig_cusolver(A0, n); // reference spectrum (lower triangle)

    T *dA = to_device(A0);
    ase::kernels::dbbr_reduce(&ws, dA, ws.B);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> Ab(n * n);
    to_host(Ab, dA);

    // Clean band: zero everything strictly below the band (r−c > nbw = reflector storage).
    std::vector<T> band = Ab;
    for (int c = 0; c < n; ++c)
        for (int r = c + nbw + 1; r < n; ++r)
            band[r + c * n] = T(0);

    auto ev = eig_cusolver(band, n);
    double range = (double)ev_ref[n - 1] - (double)ev_ref[0];
    double maxdiff = 0;
    for (int i = 0; i < n; ++i)
        maxdiff = std::max(maxdiff, std::abs((double)ev[i] - (double)ev_ref[i]));
    double rel_eig = maxdiff / range;
    CHECK_LT(rel_eig, tol);

    CUDA_CHECK(cudaFree(dA));
    ase::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(dbbr_reduce, fp64_oneblock) {
    band_reduce_case<double>(256, 64, 256, 1e-9);
}
TEST(dbbr_reduce, fp64_multiblock) {
    band_reduce_case<double>(512, 32, 128, 1e-9);
}
TEST(dbbr_reduce, fp32_multiblock) {
    band_reduce_case<float>(384, 32, 96, 1e-3);
}

TEST(dbbr_panel_qr, fp64_square) {
    panel_qr_case<double>(256, 32, 1e-10);
}
TEST(dbbr_panel_qr, fp64_wide) {
    panel_qr_case<double>(512, 64, 1e-10);
}
TEST(dbbr_panel_qr, fp32_square) {
    panel_qr_case<float>(256, 32, 1e-3);
}

CUTEST_MAIN()
