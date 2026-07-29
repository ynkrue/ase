/**
 * @file   test_backtransform.cu
 * @brief  Stage-isolation tests for the back-transform factors Q_b (BC-Back) and Q_s (SBR-Back).
 *
 * Each factor is materialized by applying its launcher to the identity, then checked against the
 * reduction invariant it must satisfy — in BOTH orientations, so the test reveals whether the
 * launcher produces Q or Qᵀ:
 *
 *   Q_s (from DBBR):  Q_sᵀ·A·Q_s == Band      (A = original symmetric, Band = DBBR output)
 *   Q_b (from BC):    Q_bᵀ·Band·Q_b == Tridiag (Tridiag from bc_chase's d,e)
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
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
#include <vector>

using namespace cutest;

// Symmetric dense matrix defined by the lower triangle of A0 (column-major, ld=n).
template <typename T> static std::vector<T> symm_from_lower(const std::vector<T> &A0, int n) {
    std::vector<T> S(n * n);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            S[i + j * n] = S[j + i * n] = A0[i + j * n];
    return S;
}

// Symmetric dense band (bandwidth b) from the lower band held in A's lower triangle.
template <typename T> static std::vector<T> band_dense(const std::vector<T> &A, int n, int b) {
    std::vector<T> S(n * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = j; i <= std::min(j + b, n - 1); ++i)
            S[i + j * n] = S[j + i * n] = A[i + j * n];
    return S;
}

// Dense tridiagonal from (d,e).
template <typename T>
static std::vector<T> tridiag_dense(const std::vector<T> &d, const std::vector<T> &e, int n) {
    std::vector<T> S(n * n, T(0));
    for (int i = 0; i < n; ++i)
        S[i + i * n] = d[i];
    for (int i = 0; i < n - 1; ++i)
        S[(i + 1) + i * n] = S[i + (i + 1) * n] = e[i];
    return S;
}

// ‖Xᵀ·M·X − R‖_F / ‖R‖_F  (X is n×n with leading dim ldX; M, R are n×n with ld=n).
template <typename T>
static double resid_TMX(const std::vector<T> &X, const std::vector<T> &M, const std::vector<T> &R,
                        int n, int ldX) {
    std::vector<T> tmp(n * n), out(n * n);
    gemm_host(tmp, M, X, n, n, n, false, false, n, ldX, n);  // tmp = M·X
    gemm_host(out, X, tmp, n, n, n, true, false, ldX, n, n); // out = Xᵀ·tmp
    return frob_diff(out, R) / std::max(1e-300, frob(R));
}

// ‖X·M·Xᵀ − R‖_F / ‖R‖_F.
template <typename T>
static double resid_XMT(const std::vector<T> &X, const std::vector<T> &M, const std::vector<T> &R,
                        int n, int ldX) {
    std::vector<T> tmp(n * n), out(n * n);
    gemm_host(tmp, M, X, n, n, n, false, true, n, ldX, n);    // tmp = M·Xᵀ
    gemm_host(out, X, tmp, n, n, n, false, false, ldX, n, n); // out = X·tmp
    return frob_diff(out, R) / std::max(1e-300, frob(R));
}

// Materialize an n×n factor by applying `apply` to the identity stored in ws.M (ld=ldu, padded).
template <typename T, typename F>
static std::vector<T> materialize(ase::SolverHandle<T> &ws, int n, F apply) {
    const int ldu = ws.ldu;
    std::vector<T> hM((size_t)ldu * n, T(0));
    for (int i = 0; i < n; ++i)
        hM[i + (size_t)i * ldu] = T(1); // identity in top n×n, padding zero
    CUDA_CHECK(cudaMemcpy(ws.M, hM.data(), hM.size() * sizeof(T), cudaMemcpyHostToDevice));
    apply(ws.M);
    CUDA_CHECK(cudaStreamSynchronize(ws.stream));
    CUDA_CHECK(cudaMemcpy(hM.data(), ws.M, hM.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return hM; // n×n factor lives in the top, ld=ldu
}

template <typename T> static void bt_case(int n, double tol) {
    const int nbw = 32, nk = 128;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = ase::handle_alloc<T>(n, nbw, nk, stream);

    std::vector<T> A0(n * n);
    fill_random(A0, 7);
    auto A = symm_from_lower(A0, n); // original symmetric matrix

    // Stage 1: DBBR  (A → band in A's lower triangle; reflectors in ws.Y / ws.W)
    T *dA = to_device(A0);
    ase::kernels::dbbr_reduce(&ws, dA, ws.B);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<T> Aband(n * n);
    to_host(Aband, dA);
    auto Band = band_dense(Aband, n, nbw);

    // Stage 2: BC  (band → tridiag d,e; reflectors in ws.U)
    ase::kernels::bc_chase(&ws, ws.B, ws.d, ws.e);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<T> d(n), e(n);
    to_host(d, ws.d);
    to_host(e, ws.e);
    auto Tri = tridiag_dense(d, e, n);

    // --- Q_s isolation:  Q_sᵀ·A·Q_s == Band ---
    auto Qs = materialize<T>(ws, n, [&](T *M) { ase::kernels::sbr_back(&ws, ws.Y, ws.W, M); });
    double qs_T = resid_TMX(Qs, A, Band, n, ws.ldu); // expects Q_s
    double qs_X = resid_XMT(Qs, A, Band, n, ws.ldu); // expects Q_sᵀ
    printf("    Q_s:  ‖Q_sᵀ·A·Q_s−B‖=%.2e   ‖Q_s·A·Q_sᵀ−B‖=%.2e\n", qs_T, qs_X);

    // --- Q_b isolation:  Q_bᵀ·Band·Q_b == Tridiag ---
    auto Qb = materialize<T>(ws, n, [&](T *M) { ase::kernels::bc_back(&ws, ws.U, M); });
    double qb_T = resid_TMX(Qb, Band, Tri, n, ws.ldu);
    double qb_X = resid_XMT(Qb, Band, Tri, n, ws.ldu);
    printf("    Q_b:  ‖Q_bᵀ·B·Q_b−T‖=%.2e   ‖Q_b·B·Q_bᵀ−T‖=%.2e\n", qb_T, qb_X);

    CHECK_LT(std::min(qs_T, qs_X), tol);
    CHECK_LT(std::min(qb_T, qb_X), tol);

    CUDA_CHECK(cudaFree(dA));
    ase::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(backtransform, fp64_n256) {
    bt_case<double>(256, 1e-9);
}
TEST(backtransform, fp64_n384) {
    bt_case<double>(384, 1e-9);
}
TEST(backtransform, fp64_n777) { // non-aligned n, 3 hop-bands
    bt_case<double>(777, 1e-9);
}
TEST(backtransform, fp64_n1024) { // 4 hop-bands
    bt_case<double>(1024, 1e-9);
}
