/**
 * @file   cublas.cu
 * @brief  Type-dispatching cuBLAS wrappers — ase::cublas namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "kernels.cuh"
#include <type_traits>

namespace ase {
namespace cublas {

template <typename T>
void gemm(SolverHandle<T> *ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,
          int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasSgemm(ws->cublas, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    else
        CUBLAS_CHECK(
            cublasDgemm(ws->cublas, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

template <typename T>
void symm(SolverHandle<T> *ws, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasSsymm(ws->cublas, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    else
        CUBLAS_CHECK(
            cublasDsymm(ws->cublas, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
}

template <typename T>
void syrk(SolverHandle<T> *ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSsyrk(ws->cublas, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    else
        CUBLAS_CHECK(cublasDsyrk(ws->cublas, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
}

template <typename T>
void syr2k(SolverHandle<T> *ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
           const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasSsyr2k(ws->cublas, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    else
        CUBLAS_CHECK(
            cublasDsyr2k(ws->cublas, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void gemm<T>(SolverHandle<T> *, cublasOperation_t, cublasOperation_t, int, int, int,  \
                          const T *, const T *, int, const T *, int, const T *, T *, int);         \
    template void symm<T>(SolverHandle<T> *, cublasSideMode_t, cublasFillMode_t, int, int,         \
                          const T *, const T *, int, const T *, int, const T *, T *, int);         \
    template void syrk<T>(SolverHandle<T> *, cublasFillMode_t, cublasOperation_t, int, int,        \
                          const T *, const T *, int, const T *, T *, int);                         \
    template void syr2k<T>(SolverHandle<T> *, cublasFillMode_t, cublasOperation_t, int, int,       \
                           const T *, const T *, int, const T *, int, const T *, T *, int);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cublas
} // namespace ase
