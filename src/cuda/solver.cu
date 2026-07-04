/**
 * @file   solver.cu
 * @brief  2-stage tridiagonalization eigensolver orchestration — single GPU.
 *
 * Pipeline: DBBR → bulge chasing → D&C (tridiagonal) → back-transform.
 * Public entry point: cuev::symm_eig_solve<T>(A, n, eval, evec, stream).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include "cuev.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <type_traits>

namespace cuev {

template <typename T>
void symm_eig_solve(T *A, int n, T *eval, T *evec, cudaStream_t stream, SolveTimer *timer) {
    using clock = std::chrono::high_resolution_clock;
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // nbw/nk: b=32 bandwidth and k=512 panel
    SolverHandle<T> ws = handle_alloc<T>(n, 32, 512, stream);

    // Optional staged timing
    cudaEvent_t e_start{}, e_dbbr{}, e_bc{}, e_dc{}, e_bt{};
    if (timer) {
        for (cudaEvent_t *e : {&e_start, &e_dbbr, &e_bc, &e_dc, &e_bt})
            CUDA_CHECK(cudaEventCreate(e));
    }
    auto wall0 = clock::now();
    if (timer) CUDA_CHECK(cudaEventRecord(e_start, stream));

    // Stage 1: full → band (DBBR), Q_s reflectors retained in ws->Y / ws->W
    kernels::dbbr_reduce(&ws, A, ws.B);
    if (timer) CUDA_CHECK(cudaEventRecord(e_dbbr, stream));

    // Stage 2: band → tridiagonal (bulge chasing), Q_b reflectors retained in ws.U
    kernels::bc_chase(&ws, ws.B, ws.d, ws.e);
    if (timer) CUDA_CHECK(cudaEventRecord(e_bc, stream));

    // Stage 3: tridiagonal D&C
    kernels::tridi_dc(&ws, ws.d, ws.e, eval, evec, A);
    if (timer) CUDA_CHECK(cudaEventRecord(e_dc, stream));

    // Stage 4: back-transform evec = Q_s · Q_b · Q_d
    kernels::back_transform(&ws, ws.Y, ws.W, ws.U, evec, timer);
    if (timer) CUDA_CHECK(cudaEventRecord(e_bt, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto wall1 = clock::now();

    if (timer) {
        CUDA_CHECK(cudaEventElapsedTime(&timer->dbbr_ms, e_start, e_dbbr));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bc_ms, e_dbbr, e_bc));
        CUDA_CHECK(cudaEventElapsedTime(&timer->dc_ms, e_bc, e_dc));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bt_ms, e_dc, e_bt));
        timer->total_ms = ms(wall0, wall1);
        for (cudaEvent_t e : {e_start, e_dbbr, e_bc, e_dc, e_bt})
            CUDA_CHECK(cudaEventDestroy(e));
    }

    handle_free(&ws);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void symm_eig_solve<float>(float *, int, float *, float *, cudaStream_t, SolveTimer *);
template void symm_eig_solve<double>(double *, int, double *, double *, cudaStream_t, SolveTimer *);
} // namespace cuev
