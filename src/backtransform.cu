/**
 * @file   backtransform.cu
 * @brief  Eigenvector back-transform — evec ← Q_s·Q_b·Q_d.
 *
 * Two-stage tridiagonalization gives A = Q_s (Q_b T Q_bᵀ) Q_sᵀ, so the eigenvectors of A are
 * evec = Q_s·Q_b·Q_d, with Q_d the tridiagonal eigenvectors from the D&C solve. The transform
 * runs in the padded buffer ws->M (ldu rows × n): load Q_d, apply Q_b then Q_s in place, copy
 * back. Applying the reflectors to Q_d directly.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 *
 *   BC-Back  (Q_b): bulge-chasing reflectors in U (column = sweep, padded ldu, unit w).
 *   SBR-Back (Q_s): DBBR WY block reflectors (I − W·Yᵀ) per panel, from W and Y.
 *
 * The fast BC-Back kernel is adapted from Wang et al. SC'25 artifact
 * (BC_kernel_computerQ_1Col_V8_10_noBandU): one warp holds a 256-row window of an M column in
 * registers and *slides* it (register shift) past the staircased reflectors staged in shared.
 * Requires b=32 and the padded U/M layout.
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

/// Partial reduction over the RED threads that span one reflector.
template <int RED> __device__ __forceinline__ double bt_reduce(double v) {
#pragma unroll
    for (int mask = RED / 2; mask > 0; mask /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    return v;
}

/**
 * @brief One pass over NC columns: apply the UTILE staged reflectors to a register window.
 *
 * The window (PT rows per lane) stays register-resident for the whole pass; the "slide up
 * one row per reflector" is done by *register renaming* — the h-loop is unrolled by PT so
 * the rotation offset u is compile-time and logical row t lives in rM[(t−u) & (PT−1)].
 * NC columns share each staged reflector load (w[]) and provide independent FMA chains to
 * hide the dot→reduce→update latency at low occupancy.
 */
template <int PT, int UTILE, int NC>
__device__ __forceinline__ void bt_pass(double *__restrict__ Mc, long ldm, long baseRow,
                                        const double *__restrict__ sU, double *__restrict__ sHead,
                                        double *__restrict__ sDone, int lane) {
    constexpr int WIN = PT * 32;
    constexpr int RED = 32 / PT;
    using V = double4_32a; // 256-bit coalesced M loads/stores, 32-byte aligned
    double rM[NC][PT];

    // load: window = bottom WIN rows of the span; sHead = the UTILE rows above it
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const V *win = reinterpret_cast<const V *>(Mc + c * ldm + baseRow + UTILE + lane * PT);
#pragma unroll
        for (int t = 0; t < PT / 4; ++t)
            reinterpret_cast<V *>(rM[c])[t] = win[t];
        for (int t = lane; t < UTILE; t += 32)
            sHead[c * UTILE + t] = Mc[c * ldm + baseRow + t];
    }
    __syncwarp();

    // reflectors h = UTILE−1 … 0, in groups of PT (u = rotation offset, compile-time)
#pragma unroll 1
    for (int hh = UTILE - PT; hh >= 0; hh -= PT) {
#pragma unroll
        for (int u = 0; u < PT; ++u) {
            const int h = hh + (PT - 1 - u);
            double w[PT]; // reflector, shared once across the NC columns
#pragma unroll
            for (int t = 0; t < PT; ++t)
                w[t] = sU[h * WIN + lane + t * 32];
            double proj[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                double p = 0.0;
#pragma unroll
                for (int t = 0; t < PT; ++t)
                    p += w[t] * rM[c][(t - u) & (PT - 1)];
                proj[c] = bt_reduce<RED>(p);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int t = 0; t < PT; ++t)
                    rM[c][(t - u) & (PT - 1)] -= 2.0 * proj[c] * w[t];
            }
            // slide: physical slot pb holds the retiring bottom row and receives the
            // incoming top row (lane−1's bottom via shuffle; lane 0 pulls from sHead).
            const int pb = (PT - 1 - u) & (PT - 1);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const double bottom = rM[c][pb];
                if (lane == 31) sDone[c * UTILE + h] = bottom;
                const double fromAbove = __shfl_up_sync(0xffffffffu, bottom, 1);
                rM[c][pb] = (lane != 0) ? fromAbove : sHead[c * UTILE + h];
            }
        }
    }
    __syncwarp(); // sDone (written by lane 31) visible before the read-out below

    // write back the slid-up window [baseRow, +WIN) and the UTILE rows that left
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        V *out = reinterpret_cast<V *>(Mc + c * ldm + baseRow + lane * PT);
#pragma unroll
        for (int t = 0; t < PT / 4; ++t)
            out[t] = reinterpret_cast<V *>(rM[c])[t];
        for (int t = lane; t < UTILE; t += 32)
            Mc[c * ldm + baseRow + WIN + t] = sDone[c * UTILE + t];
    }
}

/**
 * @brief BC-Back: M ← Q_b · M, one warp per NC columns with a register-resident sliding
 * window.
 *
 * The reduction's reflectors compose to Q_bᵀ in forward (top→bottom) order; since each Householder
 * is symmetric, Q_b is the same reflectors applied in *reverse* order. So this kernel walks sweeps
 * high→low and, within each pass, applies the UTILE staircased reflectors from h=UTILE−1 down
 * to 0 while sliding the window *up*: each step the bottom row leaves (→ sDone, written back) and a
 * fresh row enters at the top from sHead (register shuffle, no shared round-trip). Columns are
 * partitioned across blocks (first `extra` own one more); per pass the span
 * [baseRow, baseRow+WIN+UTILE) of each column is read and written.
 *
 * Geometry (template): PT = M elements per thread (window = 32·PT rows), UTILE = reflectors
 * staged in shared per pass, WARPS = warps per block, NC = columns per warp.
 *
 * @param[in]     n        matrix dimension
 * @param[in]     cols     columns owned by this block (before the +1 for large blocks)
 * @param[in]     extra    number of leading blocks that own one extra column
 * @param[in]     nsweeps  number of window sweeps (div_up(n-2, WIN))
 * @param[in]     lastU    reflector columns reaching the deepest hop-band
 * @param[in]     U        BC reflectors, ldu×n column-major
 * @param[in]     ldu      leading dim of U
 * @param[in,out] M        padded working buffer, ldm×n column-major
 * @param[in]     ldm      leading dim of M
 */
template <int PT, int UTILE, int WARPS, int NC>
__global__ void bc_back_kernel(int n, int cols, int extra, int nsweeps, int lastU,
                               const double *__restrict__ U, long ldu, double *__restrict__ M,
                               long ldm) {
    constexpr int WIN = PT * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    double *sU = reinterpret_cast<double *>(smem);        // [UTILE * WIN] reflector tile
    double *sHeads = sU + (size_t)UTILE * WIN;            // [WARPS*NC*UTILE] rows entering on top
    double *sDones = sHeads + (size_t)WARPS * NC * UTILE; // [WARPS*NC*UTILE] rows that left

    const int blk = blockIdx.x;
    if (blk < extra) {
        cols += 1;
        M += (long)blk * cols * ldm;
    } else {
        M += ((long)blk * cols + extra) * ldm;
    }
    const int lane = threadIdx.x, warp = threadIdx.y;
    double *sHead = sHeads + (size_t)warp * NC * UTILE;
    double *sDone = sDones + (size_t)warp * NC * UTILE;

    // Reverse of the forward order: sweeps high→low, passes within a sweep last→first.
    for (int sw = nsweeps - 1; sw >= 0; sw--) {
        const long baseRow0 = (long)(nsweeps - 1 - sw) * WIN;
        const int remU0 = lastU + sw * WIN;
        const int npass = (remU0 + UTILE - 1) / UTILE;
        for (int p = npass - 1; p >= 0; p--) {
            const long baseRow = baseRow0 + (long)p * UTILE;
            const long uOff = (long)p * UTILE;
            const int remU = remU0 - p * UTILE; // valid reflectors in this tile (h ≥ remU ⇒ 0)
            __syncthreads();
            // stage up to UTILE reflectors of this pass into shared (one write each; tiles
            // beyond remU — only the last pass — are zero so they act as identity reflectors).
            // U is stream-once per block → __ldcs keeps it from evicting M windows from L2.
            for (int k = warp; k < UTILE; k += WARPS) {
                if (k < remU) {
#pragma unroll
                    for (int t = 0; t < PT; t++)
                        sU[k * WIN + lane + t * 32] =
                            __ldcs(&U[(uOff + k) * ldu + baseRow + 1 + k + lane * PT + t]);
                } else {
#pragma unroll
                    for (int t = 0; t < PT; t++)
                        sU[k * WIN + lane + t * 32] = 0.0;
                }
            }
            __syncthreads();

            int col = warp * NC;
            for (; col + NC <= cols; col += WARPS * NC)
                bt_pass<PT, UTILE, NC>(M + (long)col * ldm, ldm, baseRow, sU, sHead, sDone, lane);
            for (; col < cols; ++col) // partial trailing group (at most one warp)
                bt_pass<PT, UTILE, 1>(M + (long)col * ldm, ldm, baseRow, sU, sHead, sDone, lane);
        }
    }
}

} // namespace

/// @cond INTERNAL
namespace ase {
namespace kernels {

namespace {

/// Launch one bc_back geometry (see bc_back_kernel template parameters).
template <int PT, int UTILE, int WARPS, int NC>
void bc_back_launch(AseHandle *ws, const double *U, double *M) {
    constexpr int WIN = PT * 32;
    const int n = ws->n;
    const long ldu = ws->ldu, ldm = ws->ldu;

    const int nsweeps = div_up(n - 2, WIN);
    // reflector columns reaching the deepest hop-band: s ≤ n-2-(nsweeps-1)·WIN
    const int lastU = n - 1 - (nsweeps - 1) * WIN;

    const size_t shmem = ((size_t)UTILE * WIN + 2 * (size_t)WARPS * NC * UTILE) * sizeof(double);
    CUDA_CHECK(cudaFuncSetAttribute(bc_back_kernel<PT, UTILE, WARPS, NC>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    int blocksPerSM = 0, numSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, bc_back_kernel<PT, UTILE, WARPS, NC>, 32 * WARPS, shmem));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
    const int grid = std::max(1, std::min(blocksPerSM * numSM, n));
    const int cols = n / grid;
    const int extra = n % grid;

    dim3 block(32, WARPS);
    bc_back_kernel<PT, UTILE, WARPS, NC>
        <<<grid, block, shmem, ws->stream>>>(n, cols, extra, nsweeps, lastU, U, ldu, M, ldm);
}

} // namespace

// BC-Back: M ← Q_b · M, in place on the padded buffer M (ldu×n, padding rows zeroed).
//
// Geometry tuned on A100-80GB (fp64): PT=8 (256-row window), UTILE=64, 16 warps, 2
// columns/warp — 5.7 Tflop/s at n=32k (59% of fp64 peak; sweep results in git history).
// More warps/columns hit the 163 KB shared or 64K register ceiling; smaller UTILE raises
// the (WIN+UTILE)/UTILE traffic multiplier and loses more than the occupancy gain.
void bc_back(AseHandle *ws, const double *U, double *M) {
    bc_back_launch<BC_BACK_PT, 64, 16, 2>(ws, U, M);
}

/// SBR-Back: M ← Q_s · M, in place. M is n×n (ld=ldm); WY panels applied in reverse order.
void sbr_back(AseHandle *ws, const double *Y, const double *W, double *M) {
    const int n = ws->n, b = ws->nbw;
    const int lda = ws->n, ldm = ws->ldu;
    const double one = 1.0, zero = 0.0, neg1 = -1.0;

    int jmax = 0;
    for (int j = 0; j + b < n; j += b)
        jmax = j;
    for (int j = jmax; j >= 0; j -= b) {
        const int rows = n - (j + b);
        const double *Yp = Y + (size_t)j * lda + (j + b);
        const double *Wp = W + (size_t)j * lda + (j + b);
        double *Mb = M + (j + b); // bottom row-block of M, all n columns

        // Mb ← (I − W·Yᵀ)·Mb = Mb − W·(Yᵀ·Mb)
        cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, b, n, rows, &one, Yp, lda, Mb, ldm, &zero, ws->Z,
                     b);
        cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, n, b, &neg1, Wp, lda, ws->Z, b, &one, Mb,
                     ldm);
    }
}

void back_transform(AseHandle *ws, const double *Y, const double *W, const double *U,
                    double *evec) {
    const int n = ws->n;
    const int lda = ws->n, ldm = ws->ldu;
    double *M = ws->M;

    // 1. M ← Q_d, padding rows below n zeroed for the sliding-window kernel.
    CUDA_CHECK(cudaMemsetAsync(M, 0, (size_t)ldm * n * sizeof(double), ws->stream));
    CUDA_CHECK(cudaMemcpy2DAsync(M, ldm * sizeof(double), evec, lda * sizeof(double),
                                 n * sizeof(double), n, cudaMemcpyDeviceToDevice, ws->stream));

    // 2. BC-Back: M ← Q_b · M
    bc_back(ws, U, M);

    // 3. SBR-Back: M ← Q_s · M
    sbr_back(ws, Y, W, M);

    // 4. evec ← M[:n,:]
    CUDA_CHECK(cudaMemcpy2DAsync(evec, lda * sizeof(double), M, ldm * sizeof(double),
                                 n * sizeof(double), n, cudaMemcpyDeviceToDevice, ws->stream));
}

} // namespace kernels
} // namespace ase
/// @endcond
