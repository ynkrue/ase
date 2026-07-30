/**
 * @file   solver.cu
 * @brief  2-stage tridiagonalization eigensolver orchestration — single GPU.
 *
 * Pipeline: DBBR → bulge chasing → D&C (tridiagonal) → back-transform.
 * Public entry points: ase::solve_ev (host pointers), ase::solve_ev_d (device pointers).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "ase/ase.h"
#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace ase {

namespace {

// Shared by solve_ev/solve_ev_d once ws is sized for A/eval/evec's dimension
// (handle_check already ran). A/eval/evec may be device pointers straight from the
// caller (solve_ev_d) or the handle's own staging buffers (solve_ev).
void run_pipeline(AseHandle *ws, double *A, double *eval, double *evec) {
    // Reused handles carry the previous solve's geqrf slot cursor; without this reset the
    // second solve runs past info_cap and its panel failures go unreported.
    ws->info_used = 0;
    CUDA_CHECK(cudaMemsetAsync(ws->d_info, 0, (size_t)ws->info_cap * sizeof(int), ws->stream));

    // bc_back reads the whole ldu×n extent of U, including the padding rows below n that
    // bc_chase never writes. Re-zero per solve so a reused handle cannot carry reflectors
    // from the previous problem into this one.
    CUDA_CHECK(cudaMemsetAsync(ws->U, 0, (size_t)ws->ldu * ws->n * sizeof(double), ws->stream));

    // Stage 1: full → band (DBBR), Q_s reflectors retained in ws->Y / ws->W
    kernels::dbbr_reduce(ws, A, ws->B);

    // Stage 2: band → tridiagonal (bulge chasing), Q_b reflectors retained in ws->U
    kernels::bc_chase(ws, ws->B, ws->d, ws->e);

    // Stage 3: tridiagonal D&C. A is spent by now and serves as the n×n scratch.
    kernels::tridi_dc(ws, ws->d, ws->e, eval, evec, A);

    // Stage 4: back-transform evec = Q_s · Q_b · Q_d
    kernels::back_transform(ws, ws->Y, ws->W, ws->U, evec);
}

} // namespace

void solve_ev_d(AseHandle *ws, double *A, int n, double *eval, double *evec) {
    handle_check(ws, n);
    run_pipeline(ws, A, eval, evec);
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    cusolver::geqrf_check(ws);
}

void solve_ev(AseHandle *ws, const double *A, int n, double *eval, double *evec) {
    handle_check(ws, n);
    handle_stage(ws); // device mirrors of the caller's host A/eval/evec
    const size_t es = sizeof(double);

    CUDA_CHECK(
        cudaMemcpyAsync(ws->A_stage, A, (size_t)n * n * es, cudaMemcpyHostToDevice, ws->stream));
    run_pipeline(ws, ws->A_stage, ws->eval_stage, ws->evec_stage);
    CUDA_CHECK(
        cudaMemcpyAsync(eval, ws->eval_stage, (size_t)n * es, cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(evec, ws->evec_stage, (size_t)n * n * es, cudaMemcpyDeviceToHost,
                               ws->stream));

    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    cusolver::geqrf_check(ws);
}

} // namespace ase
