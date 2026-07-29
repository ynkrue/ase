/**
 * @file   test_tridi.cu
 * @brief  Correctness tests for the tridiagonal divide-and-conquer eigensolver.
 *
 * Validates eigenpairs (eval, evec) of a random symmetric tridiagonal (d, e) directly:
 *   residual       ‖T·V − V·Λ‖_F / ‖T‖_F
 *   orthogonality  ‖Vᵀ·V − I‖_F
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include "test.h"
#include <cmath>
#include <vector>

using namespace cutest;

template <typename T>
static void tridi_case_de(std::vector<T> d, std::vector<T> e, double res_tol, double orth_tol) {
    const int n = (int)d.size();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = ase::handle_alloc<T>(n, 32, 512, stream);

    T *dd = to_device(d), *de = to_device(e), *deval, *devec, *dscr;
    CUDA_CHECK(cudaMalloc(&deval, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devec, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dscr, (size_t)n * n * sizeof(T)));

    ase::kernels::tridi_dc(&ws, dd, de, deval, devec, dscr);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> w(n), V((size_t)n * n);
    to_host(w, deval);
    to_host(V, devec);

    std::vector<T> Tm((size_t)n * n, T(0));
    for (int i = 0; i < n; ++i)
        Tm[i + (size_t)i * n] = d[i];
    for (int i = 0; i < n - 1; ++i)
        Tm[(i + 1) + (size_t)i * n] = Tm[i + (size_t)(i + 1) * n] = e[i];

    std::vector<T> TV((size_t)n * n);
    gemm_host(TV, Tm, V, n, n, n, false, false, n, n, n);
    double rnum = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double r = (double)TV[i + (size_t)j * n] - (double)w[j] * (double)V[i + (size_t)j * n];
            rnum += r * r;
        }
    CHECK_LT(std::sqrt(rnum) / frob(Tm), res_tol);

    std::vector<T> VtV((size_t)n * n);
    gemm_host(VtV, V, V, n, n, n, true, false, n, n, n);
    double onum = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double t = (double)VtV[i + (size_t)j * n] - (i == j ? 1.0 : 0.0);
            onum += t * t;
        }
    CHECK_LT(std::sqrt(onum) / std::sqrt((double)n), orth_tol);

    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(de));
    CUDA_CHECK(cudaFree(deval));
    CUDA_CHECK(cudaFree(devec));
    CUDA_CHECK(cudaFree(dscr));
    ase::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

template <typename T> static void tridi_case(int n, double res_tol, double orth_tol) {
    std::vector<T> d(n), e(n - 1);
    fill_random(d, 11);
    fill_random(e, 23);
    tridi_case_de(std::move(d), std::move(e), res_tol, orth_tol);
}

TEST(tridi_dc, fp64_leaf) {
    tridi_case<double>(48, 1e-10, 1e-10);
}
TEST(tridi_dc, fp64_onemerge) {
    tridi_case<double>(100, 1e-10, 1e-10);
}
TEST(tridi_dc, fp64_multilevel) {
    tridi_case<double>(500, 1e-9, 1e-9);
}
TEST(tridi_dc, fp32_multilevel) {
    tridi_case<float>(500, 1e-3, 1e-3);
}
// n > leaf size (512): exercises the GPU merge path across three levels
TEST(tridi_dc, fp64_gpu_merges) {
    tridi_case<double>(3000, 1e-9, 1e-9);
}
TEST(tridi_dc, fp32_gpu_merges) {
    tridi_case<float>(1500, 5e-3, 5e-3);
}
// clustered eigenvalues + small coupling: exercises heavy deflation (small-z fast
// path, Givens rotations, k << m merges)
TEST(tridi_dc, fp64_deflation) {
    const int n = 1500;
    std::vector<double> d(n), e(n - 1);
    for (int i = 0; i < n; ++i)
        d[i] = double(i % 3);
    for (int i = 0; i < n - 1; ++i)
        e[i] = 1e-6 * double(1 + (i % 5));
    tridi_case_de(std::move(d), std::move(e), 1e-10, 1e-10);
}
