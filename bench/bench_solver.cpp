/**
 * @file   bench_solver.cpp
 * @brief  Full-solver benchmark: cuev::symm_eig_solve vs cuSOLVER.
 *
 * Runs the whole pipeline (DBBR → bulge chasing → CPU D&C → back-transform) via
 * cuev::symm_eig_solve and validates the eigenpairs directly:
 *   - residual       max ‖A·V − V·Λ‖_F / ‖A‖_F
 *   - orthogonality  ‖Vᵀ·V − I‖_F
 *   - spectrum       max |λ − λ_cusolver|
 * Timed against cuSOLVER's syevd.
 *
 * Usage: cuBenchSolver [--n N]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuev.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <type_traits>
#include <vector>

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void begin(cudaStream_t s) {
        cudaEventRecord(a, s);
    }
    float end(cudaStream_t s) {
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

template <typename T> static void fill_symmetric(std::vector<T> &h, int n) {
    srand(1234);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            h[i + j * n] = h[j + i * n] = (T)(rand() % 200 - 100) / T(100);
}

// --- type-dispatching cuBLAS shims (bench is a .cpp, so no custom kernels) -------------------
template <typename T>
static void Xgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                  const T *al, const T *A, int lda, const T *B, int ldb, const T *be, T *C,
                  int ldc) {
    if constexpr (std::is_same_v<T, float>)
        cublasSgemm(h, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc);
    else
        cublasDgemm(h, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc);
}
template <typename T>
static void Xdgmm(cublasHandle_t h, cublasSideMode_t side, int m, int n, const T *A, int lda,
                  const T *x, int incx, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        cublasSdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
    else
        cublasDdgmm(h, side, m, n, A, lda, x, incx, C, ldc);
}
template <typename T>
static void Xgeam(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                  const T *al, const T *A, int lda, const T *be, const T *B, int ldb, T *C,
                  int ldc) {
    if constexpr (std::is_same_v<T, float>)
        cublasSgeam(h, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc);
    else
        cublasDgeam(h, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc);
}
template <typename T> static T Xnrm2(cublasHandle_t h, int n, const T *x, int incx) {
    T r = T(0);
    if constexpr (std::is_same_v<T, float>)
        cublasSnrm2(h, n, x, incx, &r);
    else
        cublasDnrm2(h, n, x, incx, &r);
    return r;
}
template <typename T> static T Xasum(cublasHandle_t h, int n, const T *x, int incx) {
    T r = T(0);
    if constexpr (std::is_same_v<T, float>)
        cublasSasum(h, n, x, incx, &r);
    else
        cublasDasum(h, n, x, incx, &r);
    return r;
}
template <typename T>
static void Xcopy(cublasHandle_t h, int n, const T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>)
        cublasScopy(h, n, x, incx, y, incy);
    else
        cublasDcopy(h, n, x, incx, y, incy);
}

// Validate (eval, evec) against the original symmetric A
template <typename T>
static void check_solution(cublasHandle_t cb, const T *dA, const T *d_eval, const T *d_evec, int n,
                           T *s1, T *s2, T *ddiag, double &rel_res, double &orth) {
    const T one = T(1), zero = T(0), neg1 = T(-1);

    // residual: s1 = A·V − V·Λ ;  rel = ‖s1‖_F / ‖A‖_F  (‖A‖_F = ‖λ‖₂)
    Xgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dA, n, d_evec, n, &zero, s1, n); // A·V
    Xdgmm(cb, CUBLAS_SIDE_RIGHT, n, n, d_evec, n, d_eval, 1, s2, n);                    // V·Λ
    Xgeam(cb, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one, s1, n, &neg1, s2, n, s1, n);        // A·V − V·Λ
    const double res = (double)Xnrm2(cb, n * n, s1, 1);
    const double anorm = (double)Xnrm2(cb, n, d_eval, 1);
    rel_res = res / (anorm > 0 ? anorm : 1.0);

    // orthogonality: G = Vᵀ·V ;  ‖G − I‖_F = sqrt(‖G‖_F² − 2·tr(G) + n)
    Xgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, d_evec, n, d_evec, n, &zero, s1, n);
    const double gnorm = (double)Xnrm2(cb, n * n, s1, 1);
    Xcopy(cb, n, s1, n + 1, ddiag, 1);                // diagonal of G (stride n+1)
    const double tr = (double)Xasum(cb, n, ddiag, 1); // tr(G) = Σ‖vᵢ‖² ≥ 0
    orth = std::sqrt(std::max(0.0, gnorm * gnorm - 2.0 * tr + (double)n));
}

// cuSOLVER full EVD, 64-bit Xsyevd. Returns eigenvalues and time
template <typename T>
static std::vector<double> cusolver_evd(cusolverDnHandle_t h, const std::vector<T> &hA, int n,
                                        cudaStream_t stream, float &ms, T *d_evec_out = nullptr) {
    const cudaDataType dt = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
    const auto job = CUSOLVER_EIG_MODE_VECTOR;
    const auto uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    // return {};

    T *dA, *dW;
    int *info;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));

    size_t devBytes = 0, hostBytes = 0;
    cusolverStatus_t st = cusolverDnXsyevd_bufferSize(
        h, params, job, uplo, (int64_t)n, dt, dA, (int64_t)n, dt, dW, dt, &devBytes, &hostBytes);
    if (st != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(dA);
        cudaFree(dW);
        cudaFree(info);
        cusolverDnDestroyParams(params);
        ms = -1.0f;
        return {};
    }
    void *devWork = nullptr, *hostWork = nullptr;
    CUDA_CHECK(cudaMalloc(&devWork, devBytes));
    if (hostBytes) hostWork = malloc(hostBytes);

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));
    GpuTimer t;
    t.begin(stream);
    CUSOLVER_CHECK(cusolverDnXsyevd(h, params, job, uplo, (int64_t)n, dt, dA, (int64_t)n, dt, dW,
                                    dt, devWork, devBytes, hostWork, hostBytes, info));
    ms = t.end(stream);

    std::vector<T> w(n);
    CUDA_CHECK(cudaMemcpy(w.data(), dW, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));
    if (d_evec_out) // Xsyevd overwrote dA with the eigenvectors
        CUDA_CHECK(cudaMemcpy(d_evec_out, dA, (size_t)n * n * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(info));
    CUDA_CHECK(cudaFree(devWork));
    free(hostWork);
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    return std::vector<double>(w.begin(), w.end());
}

template <typename T>
static void run_suite(cusolverDnHandle_t cusolver, cublasHandle_t cublas, int n,
                      cudaStream_t stream) {
    const char *prec = std::is_same_v<T, float> ? "fp32" : "fp64";
    const char c = std::is_same_v<T, float> ? 's' : 'd';

    std::vector<T> hA((size_t)n * n);
    fill_symmetric(hA, n);

    // cuSOLVER reference: full EVD with eigenvectors (kept in d_ref_evec for comparison).
    T *d_ref_evec;
    CUDA_CHECK(cudaMalloc(&d_ref_evec, (size_t)n * n * sizeof(T)));
    float ms_ref = 0;
    auto ev_ref = cusolver_evd<T>(cusolver, hA, n, stream, ms_ref, d_ref_evec);
    const bool have_ref = !ev_ref.empty();

    // device buffers: A (consumed by solver), eigenpair, validation scratch.
    T *dA, *d_eval, *d_evec, *s1, *s2, *ddiag;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_eval, (size_t)n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_evec, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&s1, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&s2, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&ddiag, (size_t)n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));

    // wall-clock: the pipeline mixes GPU kernels, blocking copies, and a CPU D&C.
    cuev::SolveTimer timer;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t0 = std::chrono::high_resolution_clock::now();
    cuev::symm_eig_solve<T>(dA, n, d_eval, d_evec, stream, &timer);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double ms_cuev =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
            .count();

    // symm_eig_solve overwrites A in place — restore it for the residual check.
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));
    double rel_res = 0, orth = 0;
    check_solution<T>(cublas, dA, d_eval, d_evec, n, s1, s2, ddiag, rel_res, orth);

    std::vector<T> hw(n);
    CUDA_CHECK(cudaMemcpy(hw.data(), d_eval, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));

    const char *unsup = "  unsupported (n > cuSOLVER int32 limit ~46340)";
    printf("=== full solver %s  n=%d ===\n", prec, n);
    if (have_ref) {
        printf("  cusolver_%csyevd (vectors)   %10.2f ms\n", c, ms_ref);
        printf("  cuev symm_eig_solve         %10.2f ms   (%.2fx %csyevd)\n", ms_cuev,
               ms_cuev / ms_ref, c);
    } else {
        printf("  cusolver_%csyevd (vectors) %s\n", c, unsup);
        printf("  cuev symm_eig_solve         %10.2f ms\n", ms_cuev);
    }
    cuev::solve_timer_print(timer);
    printf("  -- correctness --\n");
    printf("  residual  ‖A·V − V·Λ‖/‖A‖     %9.2e\n", rel_res);
    printf("  orthogon. ‖Vᵀ·V − I‖_F        %9.2e\n", orth);
    if (have_ref) {
        double range = ev_ref[n - 1] - ev_ref[0], maxdiff = 0;
        for (int i = 0; i < n; ++i)
            maxdiff = std::max(maxdiff, std::fabs((double)hw[i] - ev_ref[i]));
        printf("  spectrum  max|λ − cusolver|   %9.2e   (range %.3e)\n", maxdiff, range);

        // eigenvector alignment vs cuSOLVER: |⟨vᵢ, vᵢ_ref⟩| ≈ 1 (sign-free). Exact for distinct
        // eigenvalues; clustered/degenerate λ rotate within their eigenspace, so this can read
        // large even when both bases are valid — the residual above is the robust check.
        const T one = T(1), zero = T(0);
        Xgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, d_evec, n, d_ref_evec, n, &zero, s1,
              n);
        Xcopy(cublas, n, s1, n + 1, ddiag, 1); // diagonal ⟨vᵢ, vᵢ_ref⟩
        std::vector<T> hdiag(n);
        CUDA_CHECK(cudaMemcpy(hdiag.data(), ddiag, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost));
        double max_vec_err = 0;
        for (int i = 0; i < n; ++i)
            max_vec_err = std::max(max_vec_err, std::fabs(1.0 - std::fabs((double)hdiag[i])));
        printf("  eigenvec  max|1−|⟨vᵢ,vᵢ_ref⟩||%9.2e\n", max_vec_err);
    } else {
        printf("  spectrum  (no cuSOLVER reference — not checked)\n");
    }
    printf("\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_evec));
    CUDA_CHECK(cudaFree(d_ref_evec));
    CUDA_CHECK(cudaFree(s1));
    CUDA_CHECK(cudaFree(s2));
    CUDA_CHECK(cudaFree(ddiag));
}

int main(int argc, char **argv) {
    int n = 16000;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);

    printf("cuBenchSolver  n=%d  (full EVD: eigenvalues + eigenvectors)\n\n", n);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    run_suite<double>(cusolver, cublas, n, stream);
    // run_suite<float>(cusolver, cublas, n, stream);

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
