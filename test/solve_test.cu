/**
 * @file   solve_test.cu
 * @brief  End-to-end solve_ev vs LAPACK dsyevd over a deterministic corpus.
 *
 * Inputs are generated from fixed seeds (or by construction for identity / repeated /
 * clustered eigenvalue cases). Validation is sign-agnostic: eigenvalue ordering,
 * ||A·Q − Q·Λ||, and Q orthonormality.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#include "ase/ase.h"
#include "ref.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace
{

// Reconstruct A from (Q, eval) and return the max entrywise difference from the original.
std::vector<double> reconstruct(const std::vector<double>& Q, const std::vector<double>& eval,
                                int n)
{
    std::vector<double> R((size_t)n * n, 0.0);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r)
            for (int k = 0; k < n; ++k)
                R[(size_t)c * n + r] += Q[(size_t)k * n + r] * eval[k] * Q[(size_t)k * n + c];
    return R;
}

// max|Qᵀ·Q − I| over a column-major n×n Q (ld=n).
double orthogonality_err(const std::vector<double>& Q, int n)
{
    double e = 0.0;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            double dot = -((a == b) ? 1.0 : 0.0);
            for (int k = 0; k < n; ++k)
                dot += Q[(size_t)a * n + k] * Q[(size_t)b * n + k];
            e = std::max(e, std::fabs(dot));
        }
    return e;
}

std::vector<double> make_diag(int n, const char* kind)
{
    std::vector<double> d(n);
    if (kind == std::string("identity")) {
        std::fill(d.begin(), d.end(), 1.0);
    } else if (kind == std::string("repeated")) {
        for (int i = 0; i < n; ++i) d[i] = i / 3; // triples → maximal deflation
    } else { // clustered: near-equal spectrum forces the secular path
        for (int i = 0; i < n; ++i) d[i] = 1.0 + 1e-6 * (i % 5);
    }
    return d;
}

void run_case(const char* name, int n, const char* kind, unsigned seed)
{
    SCOPED_TRACE(::testing::Message() << name << " (n=" << n << ")");

    std::vector<double> A((size_t)n * n);
    if (kind == std::string("random"))
        testref::gen_symmetric(A.data(), n, n, seed);
    else
        testref::gen_rotated(A.data(), make_diag(n, kind).data(), n, n, seed);

    // LAPACK reference on a copy (ascending eval, orthonormal columns)
    std::vector<double> Alap = A, ref_eval(n);
    ASSERT_EQ(testref::dsyevd(Alap.data(), n, n, ref_eval.data()), 0);

    ase::AseHandle* ws = ase::handle_alloc(0);
    std::vector<double> eval(n), evec((size_t)n * n);
    ase::solve_ev(ws, A.data(), n, eval.data(), evec.data());
    ase::handle_free(ws);

    const double scale = testref::maxabs(n, A.data());

    EXPECT_TRUE(std::is_sorted(eval.begin(), eval.end()));
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(eval[i], ref_eval[i], testref::tol_eval(scale)) << "eigenvalue " << i;

    std::vector<double> R = reconstruct(evec, eval, n);
    double res = 0.0;
    for (size_t t = 0; t < (size_t)n * n; ++t)
        res = std::max(res, std::fabs(A[t] - R[t]));
    EXPECT_LE(res / scale, testref::tol_evec(scale)) << "A·Q − Q·Λ residual";

    EXPECT_LE(orthogonality_err(evec, n), testref::tol_evec(scale)) << "QᵀQ − I";
}

TEST(solve, RandomAcrossSizes)
{
    if (!testref::has_gpu()) GTEST_SKIP(); // solver requires n ≥ DBBR_NBW (32)
    run_case("random-32", 32, "random", 1);
    run_case("random-64", 64, "random", 3);
    run_case("random-256", 256, "random", 5);
    run_case("random-512", 512, "random", 9);
    run_case("random-1024", 1024, "random", 11);
}

TEST(solve, Identity)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    run_case("identity-96", 96, "identity", 0);
}

TEST(solve, RepeatedEigenvalues)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    run_case("repeated-120", 120, "repeated", 1);
}

TEST(solve, ClusteredEigenvalues)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    run_case("clustered-128", 128, "clustered", 2);
}

} // namespace
