/**
 * @file   kernels.cuh
 * @brief  Custom kernel launchers and cuBLAS/cuSOLVER wrappers for ASE.
 *
 * Three sections:
 *   ase::kernels   Custom GPU kernel launchers (dbbr_*, bc_*, bt_*)
 *   ase::cublas    cuBLAS wrappers   — all take AseHandle*
 *   ase::cusolver  cuSOLVER wrappers — all take AseHandle*
 *
 * All matrices are column-major, double precision throughout.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "common.h"
#include "handle.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace ase
{

// =============================================================================
// ase::kernels — custom GPU kernel launchers
// =============================================================================
namespace kernels
{

// -----------------------------------------------------------------------------
// DBBR (double-blocking band reduction)
// -----------------------------------------------------------------------------
/**
 * @brief Repack a band matrix into contiguous symmetric storage.
 *
 * Reads the lower band from full n×n storage into a packed band with
 * leading dim 2b: B[(i−j) + j·2b] = A[i,j], j ≤ i ≤ j+b. Rows b+1..2b-1
 * are zeroed to hold the transient bulge during bc_chase.
 *
 * @param[in]     ws    solver handle (stream)
 * @param[in]     A     n×n band matrix (lower band read)
 * @param[out]    B     packed band buffer, 2b×n column-major
 * @param[in]     n     matrix dimension
 * @param[in]     b     bandwidth
 */
void dbbr_pack(AseHandle* ws, const double* A, double* B, int n, int b);

/**
 * @brief Panel QR + block reflector for one DBBR panel.
 *
 * Runs geqrf on the panel and extracts the explicit unit-trapezoidal reflectors
 * into Y, and forms the b×b block factor T into ws->Tri. The Householder
 * scalars τ are staged in ws->tau.
 *      H₀·H₁···Hₖ  =  I − Y·T·Yᵀ    with Y = [v₀, v₁, …, vₖ]
 *                                   and T = upper-triangular correction.
 *
 * @param[in,out]  ws   solver handle (cusolver, geqrf scratch, tau, Tri)
 * @param[in,out] A     rows × b panel into the working matrix upper triangle ← R,
 *                      strictly-lower ← packed Householder vectors
 * @param[out]    Y     rows × b reflectors (panel of ws->Y)
 * @param[in]     rows  number of rows in the panel (n − j − b)
 * @param[in]     b     panel width (bandwidth)
 */
void dbbr_panel_qr(AseHandle* ws, double* A, double* Y, int rows, int b);

/**
 * @brief Full DBBR band reduction: symmetric A → band form (bandwidth nbw), in place.
 *
 * Orchestrates the panel QR + two-sided update over all panels. A's lower triangle
 * holds the input; on exit the band is in place (reflectors retained in ws->Y, ws->W).
 *
 * @param[in]     ws    solver handle
 * @param[in,out] A     n×n symmetric (lower), column-major, lda = ws->n; → band on exit
 * @param[out]    B     packed band buffer, 2b×n column-major
 */
void dbbr_reduce(AseHandle* ws, double* A, double* B);

// -----------------------------------------------------------------------------
// BC (bulge chasing)
// -----------------------------------------------------------------------------
/**
 * @brief GPU bulge chasing: band → tridiagonal (d, e).
 *
 * Persistent wavefront: one block per sweep, grid-strided over the n−2 sweeps with
 * gridDim capped to resident occupancy. Each sweep eliminates its column then chases
 * the bulge down, then hands off to adjacent sweep pipeline via ws->prog — sweep s
 * waits until sweep s−1 is ≥ 3b columns ahead.
 *
 * @param[in]     ws    solver handle (stream, prog)
 * @param[in,out] B     packed band matrix
 * @param[out]    d     diagonal of tridiagonal, length n
 * @param[out]    e     sub-diagonal of tridiagonal, length n−1
 */
void bc_chase(AseHandle* ws, double* B, double* d, double* e);

// -----------------------------------------------------------------------------
// DC (divide-and-conquer tridiagonal eigensolver)
// -----------------------------------------------------------------------------
/**
 * @brief GPU tridiagonal divide-and-conquer eigensolver.
 *
 * Cuppen D&C with the eigenvector matrix Q resident in
 * `evec` on the device throughout. The host runs only the O(m) deflation index
 * logic per merge (dlaed2 port); secular roots (dlaed4/5/6 device port), weight
 * fixup, eigenvector assembly, and the rank-1 update GEMMs all run on the GPU.
 * Leaves ≤ 512 are solved by host with *stedc.
 *
 * Uses ws->M (Q2 gather) and ws->Sdc (secular delta matrix) as device scratch.
 *
 * @param[in]     ws       solver handle
 * @param[in,out] d        diagonal
 * @param[in,out] e        sub-diagonal
 * @param[out]    eval     eigenvalues
 * @param[out]    evec     eigenvectors Q_d
 * @param[in,out] scratch  device scratch ≥ n×n
 */
void tridi_dc(AseHandle* ws, double* d, double* e, double* eval, double* evec, double* scratch);

/**
 * @brief Secular equation solve for one D&C merge (exposed for stage testing).
 *
 * GPU port of LAPACK dlaed4/dlaed5/dlaed6 (root finding, one thread per root) plus
 * dlaed3's Gu-Eisenstat weight fixup and O(k²) eigenvector assembly. Reads the
 * non-deflated eigenvalues/weights and the row-gather permutation already staged in
 * ws->dc_dlamda / ws->dc_w / ws->dc_ixc; writes the assembled rank-1-update
 * eigenvector matrix S and the merged eigenvalues in ws->dc_lam (ws->dc_info flags
 * root-finding failure). ws->Sdc and ws->dc_tau/dc_org/dc_wt are used as scratch.
 *
 * @param[in]     ws   solver handle (dc_dlamda, dc_w, dc_ixc staged; Sdc/dc_tau/
 *                     dc_org/dc_wt/dc_lam/dc_info scratch)
 * @param[in]     k    number of non-deflated roots
 * @param[in]     rho  rank-1 update scale (|2ρ|)
 * @param[out]    S    k×k assembled eigenvector matrix, column-major, ld = k
 */
void dc_secular_solve(AseHandle* ws, int k, double rho, double* S);

// -----------------------------------------------------------------------------
// BT (back-transform)
// -----------------------------------------------------------------------------
/**
 * @brief Apply the two-stage back-transform: evec ← Q_s · Q_b · Q_d.
 *
 * Uses ws->M (ldu×n padded) as working buffer:
 *   1. M ← Q_d          (evec copied in with zero padding for the bc_back kernel)
 *   2. BC-Back:  M ← Q_b · M   (fast sliding-window kernel; requires ldu-padded layout)
 *   3. SBR-Back: M ← Q_s · M   (WY block reflectors applied panel by panel, using ws->Z)
 *   4. evec ← M[:n,:]   (strip padding)
 *
 * @param[in]     ws    solver handle (stream, nbw, ldu; scratch Z and M)
 * @param[in]     Y     DBBR Householder reflectors, n×n column-major (ld=n)
 * @param[in]     W     SBR-Back companion W = Y·T, n×n column-major (ld=n)
 * @param[in]     U     BC Householder reflectors, ldu×n column-major (ld=ldu)
 * @param[in,out] evec  in: tridiagonal eigenvectors Q_d (n×n, ld=n);
 *                      out: full eigenvectors Q_s·Q_b·Q_d
 */
void back_transform(AseHandle* ws, const double* Y, const double* W, const double* U, double* evec);

/**
 * @brief BC-Back factor application: M ← Q_b · M (exposed for stage testing).
 * @param[in]     ws  solver handle
 * @param[in]     U   BC reflectors, ldu×n column-major
 * @param[in,out] M   padded working buffer ws->M, ldu×n (padding rows below n must be zero)
 */
void bc_back(AseHandle* ws, const double* U, double* M);

/**
 * @brief SBR-Back factor application: M ← Q_s · M (exposed for stage testing).
 * @param[in]     ws  solver handle
 * @param[in]     Y   DBBR reflectors, n×n column-major (ld=n)
 * @param[in]     W   SBR-Back companion W = Y·T, n×n column-major (ld=n)
 * @param[in,out] M   n×n working buffer (ld=ws->ldu)
 */
void sbr_back(AseHandle* ws, const double* Y, const double* W, double* M);

} // namespace kernels

// =============================================================================
// cuBLAS dispatching wrappers  (ase::cublas)
// =============================================================================
namespace cublas
{

/**
 * @brief General matrix-matrix multiplication (GEMM). C ← α·op(A)·op(B) + β·C.
 *
 * op(X) = X, Xᵀ, or Xᴴ depending on the value of transX.  C, A, B are column-major.
 *
 * @param[in] ws       solver handle
 * @param[in] transa   how to interpret A (op(A))
 * @param[in] transb   how to interpret B (op(B))
 * @param[in] m        rows of op(A) and C
 * @param[in] n        columns of op(B) and C
 * @param[in] k        columns of op(A) and rows of op(B)
 * @param[in] alpha    scalar multiplier for op(A)·op(B)
 * @param[in] A        matrix A, column-major, leading dimension lda
 * @param[in] lda      leading dimension of A (≥ rows of A)
 * @param[in] B        matrix B, column-major, leading dimension ldb
 * @param[in] ldb      leading dimension of B (≥ rows of B)
 * @param[in] beta     scalar multiplier for C
 * @param[in,out] C    matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc      leading dimension of C (≥ rows of C)
 */
void gemm(AseHandle* ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha,
          const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);

/**
 * @brief Symmetric matrix-matrix multiplication: C ← α·A·B + β·C (or B·A if side=right).
 *
 * A is symmetric; only the @p uplo triangle is referenced. B and C are column-major.
 *
 * @param[in] ws        solver handle
 * @param[in] side      whether A multiplies B from the left or right
 * @param[in] uplo      which triangle of A is referenced
 * @param[in] m         rows of B and C
 * @param[in] n         columns of B and C
 * @param[in] alpha     scalar multiplier for A·B
 * @param[in] A         symmetric matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] B         matrix B, column-major, leading dimension ldb
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
void symm(AseHandle* ws, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha,
          const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);

/**
 * @brief Symmetric rank-k update: C ← α·op(A)·op(A)ᵀ + β·C.
 *
 * Only the @p uplo triangle of C is referenced/written.
 *
 * @param[in] ws        solver handle
 * @param[in] uplo      which triangle of C is referenced (and where the result is written)
 * @param[in] trans     how to interpret A (op(A))
 * @param[in] n         rows and columns of C
 * @param[in] k         columns of op(A)
 * @param[in] alpha     scalar multiplier for op(A)·op(A)ᵀ
 * @param[in] A         matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
void syrk(AseHandle* ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha,
          const double* A, int lda, const double* beta, double* C, int ldc);

/**
 * @brief Symmetric rank-2k update: C ← α·op(A)·op(B)ᵀ + α·op(B)·op(A)ᵀ + β·C.
 *
 * Only the @p uplo triangle of C is referenced/written.
 * Used in DBBR trailing update: C = A, op(A)=Z, op(B)=Y, α=-1, β=1.
 *
 * @param[in] ws        solver handle
 * @param[in] uplo      which triangle of C is referenced (and where the result is written)
 * @param[in] trans     how to interpret A and B (op(A), op(B))
 * @param[in] n         rows and columns of C
 * @param[in] k         columns of op(A) and op(B)
 * @param[in] alpha     scalar multiplier
 * @param[in] A         matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] B         matrix B, column-major, leading dimension ldb
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
void syr2k(AseHandle* ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha,
           const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);

} // namespace cublas

// =============================================================================
// cuSOLVER type-dispatching wrappers  (ase::cusolver)
// =============================================================================
namespace cusolver
{

/**
 * @brief QR factorisation: A ← compact(Q·R), tau ← Householder scalars.
 *
 * @param[in]     ws      solver handle (cusolver, geqrf_buf, d_info)
 * @param[in]     m       number of rows
 * @param[in]     n       number of columns
 * @param[in,out] A       m×n matrix, column-major; overwritten with compact QR
 * @param[in]     lda     leading dimension of A
 * @param[out]    tau     Householder scalars, length min(m,n)
 */
void geqrf(AseHandle* ws, int m, int n, double* A, int lda, double* tau);

/**
 * @brief Check the info slots of every geqrf issued so far; aborts on failure.
 *
 * @param[in] ws  solver handle
 */
void geqrf_check(AseHandle* ws);

} // namespace cusolver

} // namespace ase
