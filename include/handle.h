/**
 * @file   handle.h
 * @brief  SolverHandle<T> — cuBLAS/cuSOLVER handles and scratch buffers for ASE.
 *
 * TODO: add description
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace ase {

/**
 * @brief Per-stage timings for one symm_eig_solve.
 *
 * GPU stages are measured with stream events, so timing adds no
 * host sync inside the pipeline.
 */
struct SolveTimer {
    float dbbr_ms = 0;  ///< full → band (DBBR)
    float bc_ms = 0;    ///< band → tridiagonal (bulge chasing)
    float dc_ms = 0;    ///< tridiagonal D&C
    float bt_ms = 0;    ///< back-transform (evec = Q_s·Q_b·Q_d)
    float total_ms = 0; ///< whole-solve wall-clock

    // back-transform breakdown (evec ← Q_s·Q_b·Q_d)
    float bt_copyin_ms = 0;  ///< M ← Q_d (memset + strided copy in)
    float bt_qb_ms = 0;      ///< BC-Back  M ← Q_b·M
    float bt_qs_ms = 0;      ///< SBR-Back M ← Q_s·M
    float bt_copyout_ms = 0; ///< evec ← M[:n,:] (strided copy out)
};

/// Print a SolveTimer breakdown
inline void solve_timer_print(const SolveTimer &t) {
    printf("  -- stage timings (ms) --\n");
    printf("    DBBR  full->band              %9.2f\n", t.dbbr_ms);
    printf("    BC    band->tridi             %9.2f\n", t.bc_ms);
    printf("    D&C   tridiagonal             %9.2f\n", t.dc_ms);
    printf("    BT    build M=Q_s·Q_b·Q_d     %9.2f\n", t.bt_ms);
    printf("      BT.copyin   M<-Q_d          %9.2f\n", t.bt_copyin_ms);
    printf("      BT.Qb       M<-Qb*M         %9.2f\n", t.bt_qb_ms);
    printf("      BT.Qs       M<-Qs*M         %9.2f\n", t.bt_qs_ms);
    printf("      BT.copyout  evec<-M         %9.2f\n", t.bt_copyout_ms);
    printf("    total (compute)               %9.2f\n", t.total_ms);
}

/**
 * @brief Per-solve context: library handles + pre-allocated scratch buffers.
 *
 * Created once in symm_eig_solve via handle_alloc, threaded through all stages,
 * destroyed via handle_free.
 *
 * @tparam T  float or double
 */
template <typename T> struct SolverHandle {
    int n;   ///< problem dimension
    int nbw; ///< bandwidth of banded matrix
    int nk;  ///< outer panel size
    int ldu; ///< padded leading dim for U and M
    cudaStream_t stream;

    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    int *d_info;

    // DBBR buffers
    T *Y;   ///< n*n - Householder reflectors, retained for SBR-Back
    T *Z;   ///< n*k - trailing two-sided companion (syr2k factor), transient per block
    T *tau; ///< nbw - Householder scalars
    T *Tri; ///< nbw*nbw - block reflector triangular factor T (larft output, transient)
    T *Dwk; ///< nk*nbw - panel scratch (Yᵀ·AY and deferral-correction GEMM temps)
    T *W;   ///< n*n - SBR-Back companion W = Y·T

    // BC buffers
    T *B;      ///< 2b*n - packed band (band + bulge space)
    T *U;      ///< ldu*n - BC Householder vectors (column = sweep, padded ldu; zero-filled)
    T *d;      ///< tridiagonal diagonal
    T *e;      ///< tridiagonal off-diagonal
    int *prog; ///< progress flag for BC

    // D&C buffers (Cuppen's algorithm on GPU)
    T *Sdc;       ///< n*n - per-merge secular delta matrix
    T *dc_z;      ///< coupling vector z extracted from Q rows
    T *dc_dlamda; ///< non-deflated eigenvalues (sorted), laed4 poles
    T *dc_w;      ///< deflation-adjusted z components (laed4 weights)
    T *dc_wt;     ///< recomputed Gu-Eisenstat weights
    T *dc_tau;    ///< per-root secular shift tau_j
    T *dc_lam;    ///< per-root eigenvalue lambda_j
    T *dc_cs;     ///< 2n - Givens (c,s) pairs from deflation
    int *dc_org;  ///< per-root origin index (lambda_j = dlamda[org] + tau)
    int *dc_indx; ///< column gather permutation (type-grouped)
    int *dc_ixc;  ///< S row gather (indxc)
    int *dc_ij;   ///< 2n - Givens column index pairs
    int *dc_info; ///< laed4 failure flag
    // pinned host mirrors for the small per-merge transfers
    T *h_z, *h_dlamda, *h_w, *h_lam, *h_cs; // h_cs is 2n
    int *h_indx, *h_ixc, *h_ij;             // h_ij is 2n
    void *host_pin;                         ///< backing allocation for the pinned mirrors

    // back-transform buffers
    T *M; ///< ldu*n - back-transform working buffer (padded for bc_back kernel)

    // cuSOLVER buffers
    T *geqrf_buf;
    int geqrf_lwork;

    // handle allocation
    void *pool;
    size_t pool_bytes;
};

/**
 * @brief Allocate all scratch and initialise cuBLAS/cuSOLVER handles.
 *
 * @tparam T     float or double
 * @param[in] n       root problem dimension
 * @param[in] nbw     bandwidth of banded matrix
 * @param[in] nk      outer panel size
 * @param[in] stream  CUDA stream all operations will run on
 */
template <typename T> SolverHandle<T> handle_alloc(int n, int nbw, int nk, cudaStream_t stream);

/**
 * @brief Destroy handles and free scratch.
 *
 * @tparam T  float or double
 * @param[in,out] ws  handle to destroy
 */
template <typename T> void handle_free(SolverHandle<T> *ws);

} // namespace ase
