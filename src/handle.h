/**
 * @file   handle.h
 * @brief  AseHandle — cuBLAS/cuSOLVER handles and scratch buffers for ASE.
 *
 * Internal header: defines the type that ase/ase.h forward-declares, plus the
 * fixed blocking constants.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "ase/ase.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace ase
{

/// Bandwidth of the intermediate banded matrix: DBBR output, bulge-chasing input.
/// Bounded by BC_MAX_B in bulge.cu, which sizes the bulge-chasing shared-memory window.
inline constexpr int DBBR_NBW = 32;

/// Outer panel width for the DBBR deferred trailing update (second level of blocking).
inline constexpr int DBBR_NK = 512;

// dbbr_reduce's outer/inner loop (src/dbbr.cu) walks panels of width DBBR_NBW nested
// inside blocks of width DBBR_NK; the panel positions only stay b-aligned across a block
// boundary if DBBR_NK is a multiple of DBBR_NBW.
static_assert(DBBR_NK % DBBR_NBW == 0, "DBBR_NK must be a multiple of DBBR_NBW");

/// D&C leaf size: sub-problems use host LAPACK *stedc
inline constexpr int DC_LEAF = 512;

/// bc_back sliding-window geometry; BC_BACK_PAD is the row padding U and M carry below row n.
inline constexpr int BC_BACK_PT  = 8;
inline constexpr int BC_BACK_WIN = BC_BACK_PT * 32;
inline constexpr int BC_BACK_PAD = 2 * BC_BACK_WIN;

/**
 * @brief library handles + pre-allocated device and host scratch.
 *
 * allocate with handle_alloc, pass the pointer to solve_ev, release with
 * handle_free. The handle owns raw device, pinned-host and host allocations,
 * it is never copied or stack-allocated by the caller.
 *
 */
struct AseHandle
{
    int          n;   ///< problem dimension
    int          nbw; ///< bandwidth of banded matrix
    int          nk;  ///< outer panel size
    int          ldu; ///< padded leading dim for U and M
    cudaStream_t stream;

    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;

    int* d_info;    ///< info slot for geqrf
    int  info_cap;  ///< number of slots
    int  info_used; ///< slots consumed so far

    // DBBR buffers
    double* Y;   ///< n*n - Householder reflectors, retained for SBR-Back
    double* Z;   ///< n*k - trailing two-sided companion (syr2k factor)
    double* tau; ///< nbw - Householder scalars
    double* Tri; ///< nbw*nbw - block reflector triangular factor T
    double* Dwk; ///< nk*nbw - panel scratch
    double* W;   ///< n*n - SBR-Back companion W = Y·T

    // BC buffers
    double* B;    ///< 2b*n - packed band (band + bulge space)
    double* U;    ///< ldu*n - BC Householder vectors (column = sweep, padded ldu; zero-filled)
    double* d;    ///< tridiagonal diagonal
    double* e;    ///< tridiagonal off-diagonal
    int*    prog; ///< progress flag for BC

    // D&C buffers (Cuppen's algorithm on GPU)
    double* Sdc;       ///< n*n - per-merge secular delta matrix
    double* dc_z;      ///< coupling vector z extracted from Q rows
    double* dc_dlamda; ///< non-deflated eigenvalues (sorted), laed4 poles
    double* dc_w;      ///< deflation-adjusted z components (laed4 weights)
    double* dc_wt;     ///< recomputed Gu-Eisenstat weights
    double* dc_tau;    ///< per-root secular shift tau_j
    double* dc_lam;    ///< per-root eigenvalue lambda_j
    double* dc_cs;     ///< 2n - Givens (c,s) pairs from deflation
    int*    dc_org;    ///< per-root origin index (lambda_j = dlamda[org] + tau)
    int*    dc_indx;   ///< column gather permutation (type-grouped)
    int*    dc_ixc;    ///< S row gather (indxc)
    int*    dc_ij;     ///< 2n - Givens column index pairs
    int*    dc_info;   ///< laed4 failure flag
    // pinned host mirrors for the small per-merge transfers
    double *h_z, *h_dlamda, *h_w, *h_lam, *h_cs; // h_cs is 2n
    int *   h_indx, *h_ixc, *h_ij;               // h_ij is 2n
    double* h_d;                                 ///< n - tridiagonal diagonal, host copy
    double* h_e;                                 ///< n - tridiagonal off-diagonal, host copy
    int*    h_info;                              ///< info_cap - geqrf info slots, host mirror
    double* h_leafQ;  ///< n*min(n,DC_LEAF) - D&C leaf eigenvector staging (host stedc → device)
    void*   host_pin; ///< backing allocation for the pinned mirrors

    // Pageable host scratch for the D&C driver.
    // The LAPACK leaf workspaces are sliced per leaf.
    int*    h_indxq;     ///< n - per-block eigenvalue permutation
    int*    h_iwk;       ///< 3n - dlaed2 index scratch
    int*    h_part;      ///< 2n+2 - D&C sub-problem partition
    size_t* h_qoff;      ///< 2n+2 - per-leaf offsets into h_leafQ
    double* h_leafwork;  ///< leafQ_len + 5n - per-leaf LAPACK stedc work
    int*    h_leafiwork; ///< 8n - per-leaf LAPACK stedc iwork
    void*   host_buf;    ///< backing allocation for the pageable host scratch

    // back-transform buffers
    double* M; ///< ldu*n - back-transform working buffer (padded)

    // cuSOLVER buffers
    double* geqrf_buf;
    int     geqrf_lwork;

    // Device staging for the host-pointer solve_ev() (host wrapper)
    void*   stage_pool; ///< backing allocation
    double* A_stage;    ///< n*n - device mirror of the caller's host A
    double* eval_stage; ///< n   - device mirror of the caller's host eval
    double* evec_stage; ///< n*n - device mirror of the caller's host evec

    // handle allocation
    void*  pool;
    size_t pool_bytes;
};

/**
 * @brief Ensure @p ws's workspace is sized for @p n, resizing only if necessary.
 *
 * No-op if @p ws is already sized for @p n. Otherwise frees any existing workspace
 * and reallocates it for the new size. If a caller genuinely interleaves two problem
 * sizes on a hot path, it should use two handles instead of resizing one repeatedly.
 *
 * @param[in,out] ws  handle to resize
 * @param[in]     n   root problem dimension; must be > 0
 */
void handle_check(AseHandle* ws, int n);

/**
 * @brief Ensure @p ws's solve_ev staging buffers exist. Call only after handle_check.
 *
 * No-op once allocated. Only solve_ev needs this.
 *
 * @param[in,out] ws  handle already sized for the target dimension
 */
void handle_stage(AseHandle* ws);

} // namespace ase
