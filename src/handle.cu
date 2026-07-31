/**
 * @file   handle.cu
 * @brief  AseHandle allocation and teardown — cuBLAS/cuSOLVER init + scratch sizing.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "ase/ase.h"
#include "common.h"
#include "handle.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ase {

namespace {

struct Pool {
    size_t off = 0;
    size_t take(size_t bytes) {
        const size_t o = off;
        off += (bytes + 255) & ~size_t(255);
        return o;
    }
};

struct Layout {
    size_t Y, Z, tau, Tri, Dwk, W, M, B, U, d, e, prog, Sdc, geqrf, dcT, dcI, info;
    size_t total;
};

Layout make_layout(int n, int nbw, int nk, int ldu, int lwork, int ninfo) {
    const size_t s = sizeof(double);
    Pool p;
    Layout L{};
    L.Y = p.take((size_t)n * n * s);
    L.Z = p.take((size_t)n * nk * s);
    L.tau = p.take((size_t)nbw * s);
    L.Tri = p.take((size_t)nbw * nbw * s);
    L.Dwk = p.take((size_t)nk * nbw * s);
    L.W = p.take((size_t)n * n * s);
    L.M = p.take((size_t)ldu * n * s);
    L.B = p.take((size_t)(2 * nbw) * n * s);
    L.U = p.take((size_t)ldu * n * s);
    L.d = p.take((size_t)n * s);
    L.e = p.take((size_t)n * s);
    L.prog = p.take((size_t)n * sizeof(int));
    L.Sdc = p.take((size_t)n * n * s);
    L.geqrf = p.take((size_t)lwork * s);
    // 6 length-n double vectors + one 2n (cs); 3 length-n int + one 2n (ij) + info
    L.dcT = p.take((size_t)8 * n * s);
    L.dcI = p.take((size_t)(5 * n + 1) * sizeof(int));
    L.info = p.take((size_t)ninfo * sizeof(int));
    L.total = p.off;
    return L;
}

/// Largest leaf the D&C can produce for this n; also bounds the per-leaf LAPACK workspace.
int dc_leaf_size(int n) {
    return std::min(n, DC_LEAF);
}

/// Number of dbbr_panel_qr calls, matching the panel loop in dbbr_reduce.
int geqrf_count(int n, int nbw) {
    int c = 0;
    for (int j = 0; j + nbw < n; j += nbw)
        ++c;
    return c;
}

/// cuSOLVER 12.x geqrf workspace is not monotonic in m, so every panel height
/// DBBR will use is queried.
int geqrf_lwork_max(cusolverDnHandle_t h, int n, int nbw) {
    double *dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, (size_t)n * nbw * sizeof(double)));

    int lwork = 0;
    for (int j = 0; j + nbw < n; j += nbw) {
        const int rows = n - j - nbw;
        int w = 0;
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(h, rows, nbw, dummy, n, &w));
        lwork = std::max(lwork, w);
    }

    CUDA_CHECK(cudaFree(dummy));
    return lwork;
}

} // namespace

size_t handle_workspace_bytes(int n) {
    cusolverDnHandle_t h = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&h));
    const int lwork = geqrf_lwork_max(h, n, DBBR_NBW);
    CUSOLVER_CHECK(cusolverDnDestroy(h));
    int ldu = ((n + BC_BACK_PAD + 3) / 4) * 4;

    return make_layout(n, DBBR_NBW, DBBR_NK, ldu, lwork, geqrf_count(n, DBBR_NBW) + 1).total;
}

AseHandle *handle_alloc(cudaStream_t stream) {
    auto *ws = new AseHandle{};
    ws->stream = stream;
    CUBLAS_CHECK(cublasCreate(&ws->cublas));
    CUBLAS_CHECK(cublasSetStream(ws->cublas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&ws->cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(ws->cusolver, stream));
    return ws;
}

void handle_check(AseHandle *ws, int n) {
    if (n <= 0) {
        fprintf(stderr, "ase: invalid problem dimension n = %d (must be > 0)\n", n);
        exit(EXIT_FAILURE);
    }
    if (ws->n == n) return; // cached

    if (ws->n != 0) {
        // Resizing pool (free). cuBLAS/cuSOLVER contexts are untouched.
        CUDA_CHECK(cudaFree(ws->pool));
        CUDA_CHECK(cudaFreeHost(ws->host_pin));
        std::free(ws->host_buf);
        CUDA_CHECK(cudaFree(ws->stage_pool));
        ws->stage_pool = nullptr;
    }

    const int nbw = DBBR_NBW;
    const int nk = DBBR_NK;

    ws->n = n;
    ws->nbw = nbw;
    ws->nk = nk;
    ws->ldu = ((n + BC_BACK_PAD + 3) / 4) * 4; // padded for double4_32a alignment

    ws->geqrf_lwork = geqrf_lwork_max(ws->cusolver, n, nbw);
    ws->info_cap = geqrf_count(n, nbw) + 1;
    ws->info_used = 0;

    const Layout L = make_layout(n, nbw, nk, ws->ldu, ws->geqrf_lwork, ws->info_cap);
    ws->pool_bytes = L.total;
    CUDA_CHECK(cudaMalloc(&ws->pool, ws->pool_bytes));

    auto base = (uint8_t *)ws->pool;
    CUDA_CHECK(cudaMemset(base + L.U, 0, (size_t)ws->ldu * n * sizeof(double))); // U padding
    CUDA_CHECK(cudaMemset(base + L.info, 0, (size_t)ws->info_cap * sizeof(int)));

    ws->Y = (double *)(base + L.Y);
    ws->Z = (double *)(base + L.Z);
    ws->tau = (double *)(base + L.tau);
    ws->Tri = (double *)(base + L.Tri);
    ws->Dwk = (double *)(base + L.Dwk);
    ws->W = (double *)(base + L.W);
    ws->M = (double *)(base + L.M);
    ws->B = (double *)(base + L.B);
    ws->U = (double *)(base + L.U);
    ws->d = (double *)(base + L.d);
    ws->e = (double *)(base + L.e);
    ws->prog = (int *)(base + L.prog);
    ws->Sdc = (double *)(base + L.Sdc);
    ws->geqrf_buf = (double *)(base + L.geqrf);
    ws->d_info = (int *)(base + L.info);

    double *dcT = (double *)(base + L.dcT);
    ws->dc_z = dcT + 0 * (size_t)n;
    ws->dc_dlamda = dcT + 1 * (size_t)n;
    ws->dc_w = dcT + 2 * (size_t)n;
    ws->dc_wt = dcT + 3 * (size_t)n;
    ws->dc_tau = dcT + 4 * (size_t)n;
    ws->dc_lam = dcT + 5 * (size_t)n;
    ws->dc_cs = dcT + 6 * (size_t)n; // 2n
    int *dcI = (int *)(base + L.dcI);
    ws->dc_org = dcI + 0 * (size_t)n;
    ws->dc_indx = dcI + 1 * (size_t)n;
    ws->dc_ixc = dcI + 2 * (size_t)n;
    ws->dc_ij = dcI + 3 * (size_t)n; // 2n
    ws->dc_info = dcI + 5 * (size_t)n;

    // Pinned host staging
    const int leaf = dc_leaf_size(n);
    // Capacity of h_leafQ in doubles
    const size_t leafQ_len = (size_t)n * leaf;
    const size_t pinT =
        ((size_t)6 * n + 2 * (size_t)n + leafQ_len) * sizeof(double); // merge + d,e + leafQ
    const size_t pinI = ((size_t)4 * n + ws->info_cap) * sizeof(int); // indx, ixc, ij(2n), info
    CUDA_CHECK(cudaMallocHost(&ws->host_pin, pinT + pinI));
    double *hT = (double *)ws->host_pin;
    ws->h_z = hT + 0 * (size_t)n;
    ws->h_dlamda = hT + 1 * (size_t)n;
    ws->h_w = hT + 2 * (size_t)n;
    ws->h_lam = hT + 3 * (size_t)n;
    ws->h_cs = hT + 4 * (size_t)n; // 2n
    ws->h_d = hT + 6 * (size_t)n;
    ws->h_e = hT + 7 * (size_t)n;
    ws->h_leafQ = hT + 8 * (size_t)n;
    int *hI = (int *)(ws->h_leafQ + leafQ_len);
    ws->h_indx = hI + 0 * (size_t)n;
    ws->h_ixc = hI + 1 * (size_t)n;
    ws->h_ij = hI + 2 * (size_t)n; // 2n
    ws->h_info = hI + 4 * (size_t)n;

    const size_t hostD = (leafQ_len + 5 * (size_t)n) * sizeof(double);
    const size_t hostI = ((size_t)6 * n + 2 + 8 * (size_t)n) * sizeof(int);
    const size_t hostS = ((size_t)2 * n + 2) * sizeof(size_t);
    ws->host_buf = std::malloc(hostD + hostI + hostS);
    if (!ws->host_buf) {
        fprintf(stderr, "ase: host scratch allocation of %zu bytes failed\n",
                hostD + hostI + hostS);
        exit(EXIT_FAILURE);
    }
    auto *hb = (uint8_t *)ws->host_buf;
    ws->h_leafwork = (double *)hb;
    ws->h_qoff = (size_t *)(hb + hostD);
    int *hJ = (int *)(hb + hostD + hostS);
    ws->h_indxq = hJ + 0 * (size_t)n;         // n
    ws->h_iwk = hJ + 1 * (size_t)n;           // 3n
    ws->h_part = hJ + 4 * (size_t)n;          // 2n+2
    ws->h_leafiwork = hJ + 6 * (size_t)n + 2; // 8n
}

void handle_stage(AseHandle *ws) {
    if (ws->stage_pool) return;

    const size_t n = (size_t)ws->n;
    const size_t doubles = 2 * n * n + n;
    CUDA_CHECK(cudaMalloc(&ws->stage_pool, doubles * sizeof(double)));

    auto *b = (double *)ws->stage_pool;
    ws->A_stage = b;
    ws->evec_stage = b + n * n;
    ws->eval_stage = b + 2 * n * n;
}

void handle_free(AseHandle *ws) {
    if (!ws) return;
    CUBLAS_CHECK(cublasDestroy(ws->cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(ws->cusolver));
    CUDA_CHECK(cudaFree(ws->pool));
    CUDA_CHECK(cudaFree(ws->stage_pool));
    CUDA_CHECK(cudaFreeHost(ws->host_pin));
    std::free(ws->host_buf);
    delete ws;
}

} // namespace ase
