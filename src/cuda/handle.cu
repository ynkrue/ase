/**
 * @file   handle.cu
 * @brief  SolverHandle allocation and teardown — cuBLAS/cuSOLVER init + scratch sizing.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include <algorithm>
#include <type_traits>

namespace cuev {

template <typename T> SolverHandle<T> handle_alloc(int n, int nbw, int nk, cudaStream_t stream) {
    SolverHandle<T> ws{};
    ws.n = n;
    ws.nbw = nbw;
    ws.nk = nk;
    ws.ldu = ((n + 512 + 3) / 4) * 4; // pad + round to 4 for aligned double4/float4 in bc_back
    ws.stream = stream;

    CUBLAS_CHECK(cublasCreate(&ws.cublas));
    CUBLAS_CHECK(cublasSetStream(ws.cublas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&ws.cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(ws.cusolver, stream));
    CUDA_CHECK(cudaMalloc(&ws.d_info, sizeof(int)));

    // Workspace query — newer cuSOLVER (12.x+) requires a non-null A pointer
    // even for the buffer-size query, so allocate a temporary dummy.
    // cuSOLVER 12.x has NON-MONOTONIC workspace in m: smaller m can request
    // *more* workspace (up to ~50M elements for m=nbw vs 82K for m=n at nbw=32),
    // so we query both extremes and take the max.
    T *geqrf_dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&geqrf_dummy, (size_t)n * nbw * sizeof(T)));
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(
            cusolverDnSgeqrf_bufferSize(ws.cusolver, n, nbw, geqrf_dummy, n, &ws.geqrf_lwork));
        int lwork_small = 0;
        CUSOLVER_CHECK(
            cusolverDnSgeqrf_bufferSize(ws.cusolver, nbw, nbw, geqrf_dummy, nbw, &lwork_small));
        if (lwork_small > ws.geqrf_lwork) ws.geqrf_lwork = lwork_small;
    } else {
        CUSOLVER_CHECK(
            cusolverDnDgeqrf_bufferSize(ws.cusolver, n, nbw, geqrf_dummy, n, &ws.geqrf_lwork));
        int lwork_small = 0;
        CUSOLVER_CHECK(
            cusolverDnDgeqrf_bufferSize(ws.cusolver, nbw, nbw, geqrf_dummy, nbw, &lwork_small));
        if (lwork_small > ws.geqrf_lwork) ws.geqrf_lwork = lwork_small;
    }
    CUDA_CHECK(cudaFree(geqrf_dummy));

    // Pool layout
    auto align_up = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    const size_t s = sizeof(T);

    size_t off = 0;
    size_t off_Y = off;
    off += align_up((size_t)n * n * s);
    size_t off_Z = off;
    off += align_up((size_t)n * nk * s);
    size_t off_tau = off;
    off += align_up((size_t)nbw * s);
    size_t off_Tri = off;
    off += align_up((size_t)nbw * nbw * s);
    size_t off_Dwk = off;
    off += align_up((size_t)nk * nbw * s);
    size_t off_W = off;
    off += align_up((size_t)n * n * s);
    size_t off_M = off;
    off += align_up((size_t)ws.ldu * n * s);
    size_t off_B = off;
    off += align_up((size_t)(2 * nbw) * n * s);
    size_t off_U = off;
    off += align_up((size_t)ws.ldu * n * s);
    size_t off_d = off;
    off += align_up((size_t)n * s);
    size_t off_e = off;
    off += align_up((size_t)n * s);
    size_t off_prog = off;
    off += align_up((size_t)n * sizeof(int));
    size_t off_Sdc = off;
    off += align_up((size_t)n * n * s);
    size_t off_geqrf = off;
    off += align_up((size_t)ws.geqrf_lwork * s);
    // D&C small staging: 6 length-n T vectors + one 2n (cs), 3 length-n int + one 2n (ij) + info
    size_t off_dcT = off;
    off += align_up((size_t)8 * n * s);
    size_t off_dcI = off;
    off += align_up((size_t)(5 * n + 1) * sizeof(int));
    ws.pool_bytes = off;

    CUDA_CHECK(cudaMalloc(&ws.pool, ws.pool_bytes));
    // zeroed out padding for U and M
    CUDA_CHECK(cudaMemset((uint8_t *)ws.pool + off_U, 0, align_up((size_t)ws.ldu * n * s)));

    auto base = (uint8_t *)ws.pool;
    ws.Y = (T *)(base + off_Y);
    ws.Z = (T *)(base + off_Z);
    ws.tau = (T *)(base + off_tau);
    ws.Tri = (T *)(base + off_Tri);
    ws.Dwk = (T *)(base + off_Dwk);
    ws.W = (T *)(base + off_W);
    ws.M = (T *)(base + off_M);
    ws.B = (T *)(base + off_B);
    ws.U = (T *)(base + off_U);
    ws.d = (T *)(base + off_d);
    ws.e = (T *)(base + off_e);
    ws.prog = (int *)(base + off_prog);
    ws.Sdc = (T *)(base + off_Sdc);
    ws.geqrf_buf = (T *)(base + off_geqrf);

    T *dcT = (T *)(base + off_dcT);
    ws.dc_z = dcT + 0 * (size_t)n;
    ws.dc_dlamda = dcT + 1 * (size_t)n;
    ws.dc_w = dcT + 2 * (size_t)n;
    ws.dc_wt = dcT + 3 * (size_t)n;
    ws.dc_tau = dcT + 4 * (size_t)n;
    ws.dc_lam = dcT + 5 * (size_t)n;
    ws.dc_cs = dcT + 6 * (size_t)n; // 2n
    int *dcI = (int *)(base + off_dcI);
    ws.dc_org = dcI + 0 * (size_t)n;
    ws.dc_indx = dcI + 1 * (size_t)n;
    ws.dc_ixc = dcI + 2 * (size_t)n;
    ws.dc_ij = dcI + 3 * (size_t)n; // 2n
    ws.dc_info = dcI + 5 * (size_t)n;

    // pinned host mirrors for the per-merge O(n) transfers
    const size_t pinT = (size_t)6 * n * s;           // z, dlamda, w, lam, cs(2n)
    const size_t pinI = (size_t)4 * n * sizeof(int); // indx, ixc, ij(2n)
    CUDA_CHECK(cudaMallocHost(&ws.host_pin, pinT + pinI));
    T *hT = (T *)ws.host_pin;
    ws.h_z = hT + 0 * (size_t)n;
    ws.h_dlamda = hT + 1 * (size_t)n;
    ws.h_w = hT + 2 * (size_t)n;
    ws.h_lam = hT + 3 * (size_t)n;
    ws.h_cs = hT + 4 * (size_t)n; // 2n
    int *hI = (int *)(hT + 6 * (size_t)n);
    ws.h_indx = hI + 0 * (size_t)n;
    ws.h_ixc = hI + 1 * (size_t)n;
    ws.h_ij = hI + 2 * (size_t)n; // 2n

    return ws;
}

template <typename T> void handle_free(SolverHandle<T> *ws) {
    CUBLAS_CHECK(cublasDestroy(ws->cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(ws->cusolver));
    CUDA_CHECK(cudaFree(ws->d_info));
    CUDA_CHECK(cudaFree(ws->pool));
    CUDA_CHECK(cudaFreeHost(ws->host_pin));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template SolverHandle<float> handle_alloc<float>(int, int, int, cudaStream_t);
template SolverHandle<double> handle_alloc<double>(int, int, int, cudaStream_t);
template void handle_free<float>(SolverHandle<float> *);
template void handle_free<double>(SolverHandle<double> *);

} // namespace cuev
