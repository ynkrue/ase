/**
 * @file   ref.hpp
 * @brief  Shared test helpers — LAPACK reference oracles and deterministic inputs.
 *
 * Ground truth for every ASE stage test is LAPACK (already a build dependency), driven
 * through its C interface (LAPACKE). Inputs are generated from fixed seeds so a run is
 * reproducible. All matrices are column-major with an explicit leading dimension.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#pragma once

namespace testref {

/// True if at least one CUDA device is present; GPU suites skip otherwise (lets CI's CPU
/// runner build and run the binary without a device).
bool has_gpu();

/// Random symmetric matrix with entries in (-1, 1]. Only the strict lower triangle is
/// written; the upper is reflected. Fixed by @p seed. Column-major, leading dim @p ld.
void gen_symmetric(double* A, int n, int ld, unsigned seed);

/// Identity matrix I_n.
void gen_identity(double* A, int n, int ld);

/// A = U·diag(@p diag)·Uᵀ with U a fixed-seed random orthogonal matrix. Builds any
/// symmetric matrix with prescribed eigenvalues (fills repeated / clustered cases).
void gen_rotated(double* A, const double* diag, int n, int ld, unsigned seed);

/// LAPACK dsyevd on the symmetric A (lower triangle read). Overwrites A with the
/// orthonormal eigenvectors (columns) and returns the eigenvalues, ascending, in @p eval.
/// Returns the LAPACK info (0 on success).
int dsyevd(double* A, int n, int ld, double* eval);

/// max|entry| of the n×n matrix, column-major with leading dim @p n.
double maxabs(int n, const double* M);

/// Eigenvalue pass tolerance: 1e-11 · max(scale, 1).
double tol_eval(double scale);
/// Eigenvector/residual tolerance: 1e-9 · max(scale, 1).
double tol_evec(double scale);

} // namespace testref
