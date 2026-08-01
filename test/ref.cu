/**
 * @file   ref.cu
 * @brief  LAPACK reference oracles + deterministic generators (see ref.hpp).
 *
 * f77 LAPACK entry points are declared extern "C" here, mirroring src/tridi.cu.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#include "ref.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <vector>

extern "C" void dsyevd_(const char* jobz, const char* uplo, const int* n, double* a,
                        const int* lda, double* w, double* work, const int* lwork, int* iwork,
                        const int* liwork, int* info);
extern "C" void dgeqrf_(const int* m, const int* n, double* a, const int* lda, double* tau,
                        double* work, const int* lwork, int* info);
extern "C" void dorgqr_(const int* m, const int* n, const int* k, double* a, const int* lda,
                        const double* tau, double* work, const int* lwork, int* info);

namespace testref
{

bool has_gpu()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void gen_symmetric(double* A, int n, int ld, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int c = 0; c < n; ++c)
        for (int r = c; r < n; ++r) { // lower triangle incl. diagonal (row r, col c)
            const double v = u(rng);
            A[(size_t)c * ld + r] = v; // (r, c) lower
            A[(size_t)r * ld + c] = v; // (c, r) upper — mirror, keeps A symmetric
        }
}

void gen_identity(double* A, int n, int ld)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[(size_t)j * ld + i] = (i == j) ? 1.0 : 0.0;
}

void gen_rotated(double* A, const double* diag, int n, int ld, unsigned seed)
{
    // U from the QR of a fixed-seed normal matrix; then A = U·diag·Uᵀ.
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> U((size_t)n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            U[(size_t)j * n + i] = g(rng);
    std::vector<double> tau(n);
    int info = 0, lq = -1;
    double wq = 0.0;
    dgeqrf_(&n, &n, U.data(), &n, tau.data(), &wq, &lq, &info);
    std::vector<double> qwork((size_t)wq);
    lq = (int)wq;
    dgeqrf_(&n, &n, U.data(), &n, tau.data(), qwork.data(), &lq, &info);
    lq = -1;
    wq = 0.0;
    dorgqr_(&n, &n, &n, U.data(), &n, tau.data(), &wq, &lq, &info);
    qwork.assign((size_t)wq, 0.0);
    lq = (int)wq;
    dorgqr_(&n, &n, &n, U.data(), &n, tau.data(), qwork.data(), &lq, &info);

    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += U[(size_t)k * n + r] * diag[k] * U[(size_t)k * n + c];
            A[(size_t)c * ld + r] = s;
        }
}

int dsyevd(double* A, int n, int ld, double* eval)
{
    const char jobz = 'V', uplo = 'L';
    int info = 0, lwork = -1, liwork = -1;
    double lw = 0.0;
    int liw = 0;
    dsyevd_(&jobz, &uplo, &n, A, &ld, eval, &lw, &lwork, &liw, &liwork, &info);
    std::vector<double> work((size_t)lw);
    std::vector<int> iwork((size_t)liw);
    lwork = (int)lw;
    liwork = (int)liw;
    dsyevd_(&jobz, &uplo, &n, A, &ld, eval, work.data(), &lwork, iwork.data(), &liwork, &info);
    return info;
}

double maxabs(int n, const double* M)
{
    double m = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            m = std::max(m, std::fabs(M[(size_t)j * n + i]));
    return m;
}

double tol_eval(double scale) { return 1e-11 * std::max(scale, 1.0); }
double tol_evec(double scale) { return 1e-9 * std::max(scale, 1.0); }

} // namespace testref
