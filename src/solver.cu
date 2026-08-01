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
#include "timing.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace ase
{

namespace
{

// Shared by solve_ev/solve_ev_d once ws is sized for A/eval/evec's dimension
// (handle_check already ran). A/eval/evec may be device pointers straight from the
// caller (solve_ev_d) or the handle's own staging buffers (solve_ev).
void solve(AseHandle* ws, double* A, double* eval, double* evec)
{
    // A reused handle carries the previous solve's geqrf cursor, and U's padded rows below n
    // are read by bc_back but never written by bc_chase, so both are re-zeroed per solve.
    ws->info_used = 0;
    CUDA_CHECK(cudaMemsetAsync(ws->d_info, 0, (size_t)ws->info_cap * sizeof(int), ws->stream));
    CUDA_CHECK(cudaMemsetAsync(ws->U, 0, (size_t)ws->ldu * ws->n * sizeof(double), ws->stream));

    // Stage 1: full → band (DBBR), Q_s reflectors retained in ws->Y / ws->W
    cudaEvent_t t0 = detail::stage_begin(ws);
    kernels::dbbr_reduce(ws, A, ws->B);
    detail::stage_end(ws, ASE_STAGE_DB, t0);

    // Stage 2: band → tridiagonal (bulge chasing), Q_b reflectors retained in ws->U
    cudaEvent_t t1 = detail::stage_begin(ws);
    kernels::bc_chase(ws, ws->B, ws->d, ws->e);
    detail::stage_end(ws, ASE_STAGE_BC, t1);

    // Stage 3: tridiagonal D&C. A is spent by now and serves as the n×n scratch.
    cudaEvent_t t2 = detail::stage_begin(ws);
    kernels::tridi_dc(ws, ws->d, ws->e, eval, evec, A);
    detail::stage_end(ws, ASE_STAGE_DC, t2);

    // Stage 4: back-transform evec = Q_s · Q_b · Q_d (split into BC-Back/SBR-Back in
    // backtransform.cu)
    kernels::back_transform(ws, ws->Y, ws->W, ws->U, evec);
}

} // namespace

void solve_ev_d(AseHandle* ws, double* A, int n, double* eval, double* evec)
{
    handle_check(ws, n);
    solve(ws, A, eval, evec);
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    cusolver::geqrf_check(ws);
}

void solve_ev(AseHandle* ws, const double* A, int n, double* eval, double* evec)
{
    handle_check(ws, n);
    handle_stage(ws); // device mirrors of the caller's host A/eval/evec
    const size_t es = sizeof(double);

    CUDA_CHECK(cudaMemcpyAsync(ws->A_stage, A, (size_t)n * n * es, cudaMemcpyHostToDevice, ws->stream));
    solve(ws, ws->A_stage, ws->eval_stage, ws->evec_stage);
    CUDA_CHECK(cudaMemcpyAsync(eval, ws->eval_stage, (size_t)n * es, cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(evec, ws->evec_stage, (size_t)n * n * es, cudaMemcpyDeviceToHost, ws->stream));

    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
    cusolver::geqrf_check(ws);
}

void ase_timing_enable(AseHandle* ws, int on) { ws->timing = (on != 0); }

void ase_timing_reset(AseHandle* ws)
{
    for (int i = 0; i < ASE_STAGE_COUNT; ++i)
        ws->timing_ms[i] = 0.0;
}

void ase_timing_read(AseHandle* ws, double* ms)
{
    for (int i = 0; i < ASE_STAGE_COUNT; ++i)
        ms[i] = ws->timing_ms[i];
}

} // namespace ase
