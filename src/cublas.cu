/**
 * @file   cublas.cu
 * @brief  cuBLAS wrappers — ase::cublas namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "kernels.cuh"

namespace ase
{
namespace cublas
{

void gemm(AseHandle* ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha,
          const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{ CUBLAS_CHECK(cublasDgemm(ws->cublas, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)); }

void symm(AseHandle* ws, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha,
          const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{ CUBLAS_CHECK(cublasDsymm(ws->cublas, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)); }

void syrk(AseHandle* ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha,
          const double* A, int lda, const double* beta, double* C, int ldc)
{ CUBLAS_CHECK(cublasDsyrk(ws->cublas, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)); }

void syr2k(AseHandle* ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha,
           const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{ CUBLAS_CHECK(cublasDsyr2k(ws->cublas, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)); }

} // namespace cublas
} // namespace ase
