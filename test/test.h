/**
 * @file   test.h
 * @brief  Minimal header-only test framework for ASE — colored, tolerance-based.
 *
 * GoogleTest-flavoured but tiny:
 *   TEST(suite, name) { ... CHECK_LT(residual, tol); }
 *   CUTEST_MAIN()
 *
 * Plus reusable host/device helpers (random fill, Frobenius norms, naive
 * column-major gemm, device round-trip).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

namespace cutest {

// --- ANSI colors -------------------------------------------------------------
constexpr const char *RESET = "\033[0m";
constexpr const char *RED = "\033[31m";
constexpr const char *GREEN = "\033[32m";
constexpr const char *CYAN = "\033[36m";
constexpr const char *BOLD = "\033[1m";

// --- test registry -----------------------------------------------------------
using TestFn = void (*)();
struct Case {
    const char *suite, *name;
    TestFn fn;
};
inline std::vector<Case> &registry() {
    static std::vector<Case> r;
    return r;
}
inline int &fail_count() { // failed checks in the current test
    static int f = 0;
    return f;
}
struct Registrar {
    Registrar(const char *s, const char *n, TestFn fn) {
        registry().push_back({s, n, fn});
    }
};

inline int run_all() {
    int passed = 0, failed = 0;
    printf("%s%s[==========]%s running %zu tests\n", BOLD, GREEN, RESET, registry().size());
    for (auto &c : registry()) {
        fail_count() = 0;
        printf("%s[ RUN      ]%s %s.%s\n", GREEN, RESET, c.suite, c.name);
        c.fn();
        if (fail_count() == 0) {
            printf("%s[       OK ]%s %s.%s\n", GREEN, RESET, c.suite, c.name);
            ++passed;
        } else {
            printf("%s[  FAILED  ]%s %s.%s (%d failed checks)\n", RED, RESET, c.suite, c.name,
                   fail_count());
            ++failed;
        }
    }
    printf("%s%s[==========]%s %s%d passed%s, %s%d failed%s\n", BOLD, GREEN, RESET, GREEN, passed,
           RESET, failed ? RED : GREEN, failed, RESET);
    return failed;
}

// --- reusable helpers --------------------------------------------------------
template <typename T> void fill_random(std::vector<T> &h, unsigned seed = 1) {
    srand(seed);
    for (auto &x : h)
        x = (T)(rand() % 2000 - 1000) / T(1000);
}

template <typename T> double frob(const std::vector<T> &a) {
    double s = 0;
    for (auto x : a)
        s += (double)x * (double)x;
    return std::sqrt(s);
}

template <typename T> double frob_diff(const std::vector<T> &a, const std::vector<T> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

template <typename T> T *to_device(const std::vector<T> &h) {
    T *d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T> void to_host(std::vector<T> &h, const T *d) {
    cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

/// Naive column-major gemm: C(m×n) = op(A)·op(B). tA/tB select transpose.
template <typename T>
void gemm_host(std::vector<T> &C, const std::vector<T> &A, const std::vector<T> &B, int m, int n,
               int k, bool tA, bool tB, int lda, int ldb, int ldc) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0;
            for (int l = 0; l < k; ++l) {
                T a = tA ? A[l + i * lda] : A[i + l * lda];
                T b = tB ? B[j + l * ldb] : B[l + j * ldb];
                acc += (double)a * (double)b;
            }
            C[i + j * ldc] = (T)acc;
        }
}

} // namespace cutest

// --- registration + assertion macros -----------------------------------------
#define TEST(suite, name)                                                                          \
    static void suite##_##name##_impl();                                                           \
    static cutest::Registrar suite##_##name##_reg(#suite, #name, suite##_##name##_impl);           \
    static void suite##_##name##_impl()

#define CHECK_LT(val, tol)                                                                         \
    do {                                                                                           \
        double _v = (double)(val);                                                                 \
        if (!(_v < (double)(tol))) {                                                               \
            ++cutest::fail_count();                                                                \
            printf("%s    %s = %.3e  !<  %.1e%s  (%s:%d)\n", cutest::RED, #val, _v, (double)(tol), \
                   cutest::RESET, __FILE__, __LINE__);                                             \
        }                                                                                          \
    } while (0)

#define CUTEST_MAIN()                                                                              \
    int main() {                                                                                   \
        return cutest::run_all();                                                                  \
    }
