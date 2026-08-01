/**
 * @file   ase.h
 * @brief  Public API for ASE — single-GPU symmetric dense eigensolver.
 *
 * Computes the full eigendecomposition of a real symmetric matrix in double precision
 * on one GPU, via two-stage tridiagonalization: dense → band (DBBR), band → tridiagonal
 * (bulge chasing), tridiagonal divide-and-conquer, then back-transform of the
 * eigenvectors. All matrices are column-major.
 *
 * AseHandle: handle_alloc creates the cuBLAS/cuSOLVER contexts, and its workspace,
 * which is sized lazily and cached across repeated calls at the same dimension.
 *
 * Usage example:
 *   ase::AseHandle *ws = ase::handle_alloc(stream);
 *   ase::solve_ev(ws, A, n, eval, evec);      // host pointers
 *   ase::solve_ev_d(ws, A, n, eval, evec);    // device pointers
 *   ase::handle_free(ws);
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace ase {

/**
 * @brief library handles + pre-allocated device and host scratch.
 *
 * allocate with handle_alloc, pass the pointer to solve_ev, release with
 * handle_free. The handle owns raw device, pinned-host and host allocations,
 * it is never copied or stack-allocated by the caller.
 */
struct AseHandle;

/**
 * @brief Create a new solver handle.
 *
 * Allocate cuBLAS/cuSOLVER contexts, but does not allocate the handle's
 * workspace. The first solve_ev call sizes the workspace and later calls
 * reuse as long as @p n doesn't change. Must be released with handle_free.
 *
 * @param[in] stream  CUDA stream all operations will run on
 * @return  a handle owning no workspace yet; never null (allocation failure aborts)
 */
AseHandle *handle_alloc(cudaStream_t stream);

/**
 * @brief Destroy the cuBLAS/cuSOLVER contexts, free all scratch, and delete @p ws.
 *
 * @param[in] ws  handle to destroy; null is accepted and ignored. The pointer is
 *                dangling on return.
 */
void handle_free(AseHandle *ws);

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix (host pointers).
 *
 * Copies @p A to the device, solves, and copies @p eval/@p evec back (convenience wrapper
 * around solve_ev_d) @p ws's workspace is (re)sized for @p n (if not cached).
 *
 * @param[in,out] ws    solver handle
 * @param[in]     A     n×n real symmetric matrix (column-major, host); only the lower
 *                      triangle is read. Not modified (copied to device).
 * @param[in]     n     matrix dimension (> 0)
 * @param[out]    eval  eigenvalues in ascending order, length n (host)
 * @param[out]    evec  eigenvectors as columns, n×n column-major (host);
 *                      evec[j*n + i] is component i of eigenvector j
 */
void solve_ev(AseHandle *ws, const double *A, int n, double *eval, double *evec);

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix (device pointers).
 *
 * @p ws's workspace is (re)sized for @p n (if not cached).
 *
 * @param[in,out] ws    solver handle
 * @param[in,out] A     n×n real symmetric matrix (column-major, device); only the lower
 *                      triangle is read. **Destroyed**, the solver reduces A in place and
 *                      then uses the buffer.
 * @param[in]     n     matrix dimension (> 0)
 * @param[out]    eval  eigenvalues in ascending order, length n (device)
 * @param[out]    evec  eigenvectors as columns, n×n column-major (device);
 *                      evec[j*n + i] is component i of eigenvector j
 */
void solve_ev_d(AseHandle *ws, double *A, int n, double *eval, double *evec);

/// Pipeline stages that can be timed individually: the three reduction/solve stages plus
/// the back-transform split into its two halves (BC-Back, SBR-Back).
enum ase_stage {
    ASE_STAGE_DB = 0,
    ASE_STAGE_BC,
    ASE_STAGE_DC,
    ASE_STAGE_BCBACK,
    ASE_STAGE_SBR,
    ASE_STAGE_COUNT
};

/**
 * @brief Enable (on != 0) or disable per-stage CUDA-event timing.
 *
 * Off by default; when off the solve runs without recording. When on, each stage's time
 * accumulates in the handle until ase_timing_read / ase_timing_reset.
 *
 * @param[in,out] ws  solver handle
 * @param[in]     on  nonzero to record
 */
void ase_timing_enable(AseHandle *ws, int on);

/// Zero the accumulated stage times on @p ws.
void ase_timing_reset(AseHandle *ws);

/**
 * @brief Copy the accumulated per-stage times (ms) into @p ms (ASE_STAGE_COUNT entries).
 *
 * Values are cumulative across every solve since the last reset; divide by the number of
 * solves to get per-solve averages. Zero when timing was never enabled.
 */
void ase_timing_read(AseHandle *ws, double *ms);

} // namespace ase
