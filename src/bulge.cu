/**
 * @file   bulge.cu
 * @brief  Bulge chasing kernels — band repack + band-to-tridiagonal reduction.
 *
 * Two stages (see Wang et al. PPoPP'25 §2.4/§4.2):
 *   bc_pack   full-storage band → packed band, 2b rows (extra rows hold the bulge)
 *   bc_chase  packed band → tridiagonal (d, e)
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 *
 * The chase applies a Householder per column: H·A·H on the working window,
 * eliminating one column's sub-band and spilling a bulge just below the
 * band, where it is chased down by b each hop.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include <algorithm>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

/// Largest supported bandwidth.
constexpr int BC_MAX_B = 64;
/// Threads per sweep-block (sized to the ~3b-wide per-hop window, not the band)
constexpr int BC_THREADS = 128;
/// Warps per block
constexpr int BC_NWARPS = BC_THREADS / 32;

/// Block-wide all-reduce sum via warp shuffles + one cross-warp pass (red holds BC_NWARPS
/// partials).
template <typename T> __device__ __forceinline__ T block_sum(T v, T *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    T r = (threadIdx.x < BC_NWARPS) ? red[threadIdx.x] : T(0);
    if (warp == 0) {
#pragma unroll
        for (int o = BC_NWARPS / 2; o > 0; o >>= 1)
            r += __shfl_down_sync(0xffffffffu, r, o);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    const T result = red[0];
    __syncthreads();
    return result;
}

/**
 * @brief Form the normalized Householder reflector that annihilates x[1:].
 *
 * Block-collective builds a unit-norm reflector w with H = I − 2·w·wᵀ and
 * H·x = β·e₀; β (the surviving entry, → d/e) is returned to all threads.
 * Normalized convention (‖w‖ = 1, implicit τ = 2) needs no τ array.
 * β = −sign(x[0])·‖x‖ avoids cancellation. x is overwritten with w.
 */
template <typename T> __device__ void householder(T *x, int L, T &beta, T *red) {
    const int tid = threadIdx.x;
    const T alpha = x[0];

    T local = T(0);
    for (int t = tid; t < L; t += BC_THREADS)
        local += x[t] * x[t];
    const T sumsq = block_sum(local, red);

    if (sumsq == T(0)) {
        for (int t = tid; t < L; t += BC_THREADS)
            x[t] = T(0);
        beta = T(0);
        return;
    }

    const T sign = (alpha >= T(0)) ? T(1) : T(-1);
    beta = -sign * sqrt(sumsq);
    const T v0 = alpha - beta;
    const T inv = T(1) / sqrt(v0 * v0 + (sumsq - alpha * alpha)); // 1/‖v‖

    if (tid == 0) x[0] = v0;
    __syncthreads();
    for (int t = tid; t < L; t += BC_THREADS)
        x[t] *= inv;
    __syncthreads();
}

/**
 * @brief One bulge-chasing reflector step.
 *
 * Eliminates column cq below row r0 with a Householder on rows [r0,r1], then applies
 * the two-sided similarity A ← H·A·H as a rank-2 update over the affected range
 * [cq, min(n-1, r1+b)]. Entries left of cq are already reduced and left untouched.
 * This is one iteration of Algorithm 2 from Wang et al. (the H·A·H of B_d, B_ol, B_od).
 */
template <typename T>
__device__ void bc_apply_step(T *B, T *sx, T *sw, T *sp, T *sy, T *red, int r0, int r1, int cq,
                              int n, int ldb) {
    const int tid = threadIdx.x;
    const int b = ldb / 2;
    const int L = r1 - r0 + 1;
    const int lo = cq, hi = min(n - 1, r1 + b), W = hi - lo + 1;

    // gather column cq at rows [r0,r1] → reflector w
    for (int t = tid; t < L; t += BC_THREADS)
        sx[t] = band_sym(B, r0 + t, cq, ldb);
    __syncthreads();
    T beta;
    householder(sx, L, beta, red);

    // p_i = 2·Σ_{k∈[r0,r1]} A[i,k]·w_k
    for (int idx = tid; idx < W; idx += BC_THREADS) {
        const int i = lo + idx;
        T acc = T(0);
        for (int t = 0; t < L; ++t)
            acc += band_sym(B, i, r0 + t, ldb) * sx[t];
        sp[idx] = T(2) * acc;
        sw[idx] = (i >= r0 && i <= r1) ? sx[i - r0] : T(0);
    }
    __syncthreads();

    // κ = Σ w_i·p_i ;  y_i = p_i − κ·w_i
    T kloc = T(0);
    for (int t = tid; t < L; t += BC_THREADS)
        kloc += sx[t] * sp[(r0 + t) - lo];
    const T kappa = block_sum(kloc, red);
    for (int idx = tid; idx < W; idx += BC_THREADS)
        sy[idx] = sp[idx] - kappa * sw[idx];
    __syncthreads();

    // rank-2 update A[i,j] −= w_i·y_j + y_i·w_j
    const int warp = tid >> 5, lane = tid & 31, nwarps = BC_THREADS >> 5;
    for (int j = lo + warp; j <= hi; j += nwarps) {
        const T wj = sw[j - lo], yj = sy[j - lo];
        for (int d = lane; d < 2 * b; d += 32) {
            const int i = j + d;
            if (i > hi) continue;
            const T upd = sw[i - lo] * yj + sy[i - lo] * wj;
            if (upd != T(0)) B[d + (size_t)j * ldb] -= upd;
        }
    }
    __syncthreads();
}

/**
 * @brief Bulge chasing: packed band → tridiagonal (d, e).
 *
 * Algorithm 2 (Wang et al.), persistent wavefront. One block per sweep,
 * grid-strided over the n−2 sweeps. Sweep s eliminates column s then chases
 * the bulge down by b each hop (eliminated column c = s+1, s+1+b, s+1+2b, …);
 * the bulge stays within sub-diagonal 2b-1, held by the 2b-row band — n/b hops
 * per sweep, so the chase is O(n²b).
 *
 * Wavefront handoff: before processing column c, sweep s spins until the previous
 * sweep's frontier prog[s-1] is ≥ c + 3b ahead (affected ranges, each ≤ 3b wide,
 * stay disjoint), then publishes prog[s] = c with a release fence. gridDim is capped
 * to resident occupancy so every block is co-resident. prog[s] = n + 3b on
 * completion releases the successor's tail.
 *
 * d/e are extracted by bc_extract_kernel afterwards. Each hop's reflector w is
 * stashed into U[:,s].
 */
template <typename T>
__global__ void bc_chase_kernel(T *B, T *U, int ldu, int n, int b, int *prog) {
    __shared__ T sx[BC_MAX_B];     // reflector w
    __shared__ T sw[3 * BC_MAX_B]; // w as a row-indexed vector over [lo,hi]
    __shared__ T sp[3 * BC_MAX_B]; // p = 2·A·w over the affected range
    __shared__ T sy[3 * BC_MAX_B]; // y = p − κ·w
    __shared__ T red[BC_THREADS];  // block-reduction scratch

    const int tid = threadIdx.x;
    const int ldb = 2 * b;
    const int margin = 3 * b;
    volatile int *vprog = prog;

    for (int s = blockIdx.x; s < n - 2; s += gridDim.x) {
        // c = eliminated column (step b, Algo 2).
        for (int c = s;; c = (c == s) ? s + 1 : c + b) {
            // c > s chases the bulge down by b
            int r0, r1, cq;
            if (c == s) {
                r0 = s + 1, r1 = min(s + b, n - 1), cq = s;
            } else {
                if (c + b >= n) break;
                r0 = c + b, r1 = min(c + 2 * b - 1, n - 1), cq = c;
            }

            // wait until the previous sweep is ≥ margin ahead (aquire)
            if (s > 0) {
                if (tid == 0)
                    while (vprog[s - 1] < c + margin) { /* spin */
                    }
                __syncthreads();
                __threadfence();
            }

            bc_apply_step(B, sx, sw, sp, sy, red, r0, r1, cq, n, ldb);

            // stash reflector w (sx, rows [r0,r1]) into U[:,s] for BC-Back
            for (int t = tid; t <= r1 - r0; t += BC_THREADS)
                __stcs(&U[(size_t)(r0 + t) + (size_t)s * ldu], sx[t]);

            // publish frontier (release)
            __threadfence();
            if (tid == 0) vprog[s] = c;
            __syncthreads();
        }
        __threadfence();
        if (tid == 0) vprog[s] = n + 3 * b;
        __syncthreads();
    }
}

/// Extract the tridiagonal (d, e) from the reduced packed band.
template <typename T> __global__ void bc_extract_kernel(const T *B, T *d, T *e, int n, int b) {
    const int ldb = 2 * b;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = B[(size_t)i * ldb];
    if (i < n - 1) e[i] = B[1 + (size_t)i * ldb];
}

} // namespace

namespace ase {
namespace kernels {

template <typename T> void bc_chase(SolverHandle<T> *ws, T *B, T *d, T *e) {
    int n = ws->n, b = ws->nbw;
    const int nsweeps = n - 2;
    if (nsweeps <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(ws->prog, 0xFF, (size_t)n * sizeof(int), ws->stream));

    int blocksPerSM = 0, numSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, bc_chase_kernel<T>,
                                                             BC_THREADS, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
    int wavefront = 2 * div_up(n, 3 * b);
    int grid = std::max(1, std::min(std::min(nsweeps, wavefront), blocksPerSM * numSM));
    bc_chase_kernel<<<grid, BC_THREADS, 0, ws->stream>>>(B, ws->U, ws->ldu, n, b, ws->prog);

    bc_extract_kernel<<<div_up(n, 256), 256, 0, ws->stream>>>(B, d, e, n, b);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T) template void bc_chase<T>(SolverHandle<T> *, T *, T *, T *);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace ase
