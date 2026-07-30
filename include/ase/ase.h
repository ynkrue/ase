/**
 * @file   ase.h
 * @brief  Public API for ASE — single-GPU symmetric dense eigensolver.
 *
 * Computes the full eigendecomposition of a real symmetric matrix in double precision
 * on one GPU, via two-stage tridiagonalization: dense → band (DBBR), band → tridiagonal
 * (bulge chasing), tridiagonal divide-and-conquer, then back-transform of the
 * eigenvectors. All matrices are column-major.
 *
 * Every solve goes through an AseHandle: handle_alloc creates the cuBLAS/cuSOLVER
 * contexts, and its workspace is sized lazily by the first solve_ev/solve_ev_d call
 * and cached across repeated calls at the same dimension — no per-solve allocation.
 * There is no handle-free convenience overload; the handle is not optional.
 *
 *   ase::AseHandle *ws = ase::handle_alloc(stream);
 *   ase::solve_ev(ws, A, n, eval, evec);      // host pointers
 *   ase::solve_ev_d(ws, A, n, eval, evec);    // device pointers
 *   ase::handle_free(ws);
 *
 * This is the only installed header. AseHandle is opaque — its definition, the blocking
 * constants and the kernel launchers are internal to the library (src/).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace ase {

/**
 * @brief Per-solve context: library handles + pre-allocated device and host scratch.
 *
 * Opaque by design — allocate with handle_alloc, pass the pointer around, release with
 * handle_free. The handle owns raw device, pinned-host and host allocations, so it is
 * never copied or stack-allocated by the caller.
 */
struct AseHandle;

/**
 * @brief Allocate an AseHandle: create the cuBLAS/cuSOLVER contexts.
 *
 * The handle's workspace is not sized here — it does not know the problem dimension
 * yet. The first solve_ev/solve_ev_d call sizes it; later calls reuse that sizing as
 * long as @p n doesn't change. Must be released with handle_free.
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
 * @brief Device bytes a solve at dimension @p n will allocate in its handle.
 *
 * Covers the solver workspace — what solve_ev_d needs on top of the caller's own A, eval
 * and evec. solve_ev additionally allocates device mirrors of its host arguments, a
 * further `(2n² + n) · sizeof(double)`, on its first call. Excludes host-side scratch,
 * which is not device memory.
 *
 * The blocking parameters are fixed internally, so the footprint depends only on @p n.
 * Independent of any handle — usable before one exists, to check a problem will fit.
 *
 * @param[in] n  root problem dimension (> 0)
 */
size_t handle_workspace_bytes(int n);

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix (host pointers).
 *
 * Copies @p A to the device, solves, and copies @p eval/@p evec back — a convenience
 * wrapper around solve_ev_d for callers who don't otherwise manage device memory. @p ws's
 * workspace is (re)sized for @p n if it isn't already; reusing the same handle at the same
 * @p n across repeated calls costs one H2D and two D2H copies per solve, nothing else.
 *
 * @param[in,out] ws    solver handle (any state — will be sized for @p n if needed)
 * @param[in]     A     n×n real symmetric matrix (column-major, host); only the lower
 *                      triangle is read. Not modified — the device copy is destroyed,
 *                      not @p A itself. (Hence const, unlike solve_ev_d's @p A.)
 * @param[in]     n     matrix dimension (> 0)
 * @param[out]    eval  eigenvalues in ascending order, length n (host)
 * @param[out]    evec  eigenvectors as columns, n×n column-major (host);
 *                      evec[j*n + i] is component i of eigenvector j
 */
void solve_ev(AseHandle *ws, const double *A, int n, double *eval, double *evec);

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix (device pointers).
 *
 * @p ws's workspace is (re)sized for @p n if it isn't already; a repeated call at the
 * same @p n does no allocation at all.
 *
 * @param[in,out] ws    solver handle (any state — will be sized for @p n if needed)
 * @param[in,out] A     n×n real symmetric matrix (column-major, device); only the lower
 *                      triangle is read. **Destroyed** — the solver reduces A in place and
 *                      then reuses the buffer as n×n scratch for the divide-and-conquer
 *                      stage. Copy it beforehand if the input is still needed.
 * @param[in]     n     matrix dimension (> 0)
 * @param[out]    eval  eigenvalues in ascending order, length n (device)
 * @param[out]    evec  eigenvectors as columns, n×n column-major (device);
 *                      evec[j*n + i] is component i of eigenvector j
 */
void solve_ev_d(AseHandle *ws, double *A, int n, double *eval, double *evec);

} // namespace ase
