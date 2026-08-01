/**
 * @file   dbbr.cu
 * @brief  Double-blocking band reduction kernels — panel QR, larft, custom syr2k.
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
namespace
{

/**
 * @brief Pack the lower band of n×n matrix into a 2b-row packed band.
 *
 * Column-major; sub-diagonal index (i-j) is the packed row, leading dim 2b.
 * Rows 0..b hold the band A[j..j+b, j] and rows b+1..2b-1 are zeroed.
 *
 * @param[in]  A    n×n band matrix
 * @param[out] Bp   packed band, 2b×n
 * @param[in]  n    matrix dimension
 * @param[in]  b    bandwidth
 * @param[in]  lda  leading dimension of A
 */
__global__ void bc_pack_kernel(const double* A, double* Bp, int n, int b, int lda)
{
    const int ldb = 2 * b;
    int       r   = blockIdx.x * blockDim.x + threadIdx.x;
    int       j   = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= ldb || j >= n) return;

    int i           = j + r;
    Bp[r + j * ldb] = (r <= b && i < n) ? A[i + (size_t)j * lda] : 0.0;
}

/**
 * @brief Masked copy of the packed QR panel A into the reflector buffer Y.
 *
 * Column-major, leading dimension lda for both A and Y.
 *   r <  c  →  Y[r,c] = 0          (above diagonal — R's upper part)
 *   r == c  →  Y[r,c] = 1          (unit diagonal, implicit in geqrf output)
 *   r >  c  →  Y[r,c] = A[r,c]     (essential Householder entries)
 *
 * @param[in]  A     packed panel (column-major)
 * @param[out] Y     explicit unit lower-trapezoidal reflectors (column-major)
 * @param[in]  rows  rows in the panel
 * @param[in]  cols  columns in the panel (bandwidth b)
 * @param[in]  lda   leading dimension of A and Y (= ws->n)
 */
__launch_bounds__(256) __global__ void dbbr_extract_reflectors(const double* A, double* Y, int rows, int cols, int lda)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    double v;
    if (r < c)
        v = 0.0;
    else if (r == c)
        v = 1.0;
    else
        v = A[c * lda + r];

    Y[c * lda + r] = v;
}

/**
 * @brief Form the b×b block reflector factor T.
 *
 * Given the Gram matrix G = Vᵀ·V (upper triangle) and the Householder scalars τ,
 * builds upper-triangular T with H₀···H_{b-1} = I − V·T·Vᵀ via the recurrence
 *   T[i,i]    =  τᵢ
 *   T[0:i, i] = −τᵢ · (T[0:i, 0:i] · G[0:i, i]).
 *
 * @param[in]  G    Vᵀ·V, b×b column-major, ldg = b (upper triangle referenced)
 * @param[in]  tau  Householder scalars, length b
 * @param[out] Tri  b×b block factor, column-major, ldt = b
 * @param[in]  b    panel width (≤ 64)
 */
__global__ void dbbr_larft(const double* G, const double* tau, double* Tri, int b)
{
    __shared__ double sT[64 * 64];
    const int         tid = threadIdx.x;

    for (int c = 0; c < b; ++c)
        sT[tid + c * b] = 0.0;
    __syncthreads();

    // Columns of T are built left to right; column i needs columns 0..i-1.
    for (int i = 0; i < b; ++i)
    {
        if (tid < i)
        {
            double acc = 0.0;
            for (int q = tid; q < i; ++q)
                acc += sT[tid + q * b] * G[q + i * b]; // T[p,q]·G[q,i]
            sT[tid + i * b] = -tau[i] * acc;
        }
        else if (tid == i) { sT[tid + i * b] = tau[i]; }
        __syncthreads();
    }

    for (int c = 0; c < b; ++c)
        Tri[tid + c * b] = sT[tid + c * b];
}

} // namespace

namespace ase
{
namespace kernels
{

void dbbr_pack(AseHandle* ws, const double* A, double* Bp, int n, int b)
{
    const int     lda = ws->n;
    constexpr int BX = 32, BY = 8;
    dim3          block(BX, BY);
    dim3          grid(div_up(2 * b, BX), div_up(n, BY));
    bc_pack_kernel<<<grid, block, 0, ws->stream>>>(A, Bp, n, b, lda);
}

void dbbr_panel_qr(AseHandle* ws, double* A, double* Y, int rows, int b)
{
    const int lda = ws->n;

    // QR factorization: R in upper triangle of A, Householder vectors
    // below the diagonal, scalars into ws->tau.
    cusolver::geqrf(ws, rows, b, A, lda, ws->tau);

    // extract lower-trapezoidal reflectors  A → Y
    constexpr int BX = 64, BY = 4;
    dim3          block(BX, BY);
    dim3          grid(div_up(rows, BX), div_up(b, BY));
    dbbr_extract_reflectors<<<grid, block, 0, ws->stream>>>(A, Y, rows, b, lda);

    // larft: build b×b block factor T from Y and ws->tau into ws->Tri
    //   Gram matrix G = Yᵀ·Y with syrk → ws->Dwk
    const double one = 1.0, zero = 0.0;
    cublas::syrk(ws, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, b, rows, &one, Y, lda, &zero, ws->Dwk, b);
    //   triangular recurrence → ws->Tri
    dbbr_larft<<<1, b, 0, ws->stream>>>(ws->Dwk, ws->tau, ws->Tri, b);
}

void dbbr_reduce(AseHandle* ws, double* A, double* B)
{
    int    n = ws->n, b = ws->nbw, k = ws->nk;
    int    lda      = ws->n;
    double zero     = 0.0;
    double one      = 1.0;
    double neg1     = -1.0;
    double neg_half = -0.5;

    // outer (block) loop
    for (int i = 0; i < n; i += k)
    {
        int kc         = std::min(k, n - i); // current block width
        int block_cols = 0;                  // reflector columns Y/Z accumulated so far in this block

        // inner (panel) loop
        for (int j = i; j < i + kc && j + b < n; j += b)
        {
            int pc   = std::min(b, n - j); // current panel width
            int rows = n - (j + pc);
            int off  = j * lda + (j + pc);
            int tr   = (j + pc) * lda + (j + pc);
            int zc   = (j - i) * lda + (j + pc);

            /// Panel QR factorization red panel in [Algorithm 1, Wang et al. 2025]
            kernels::dbbr_panel_qr(ws, A + off, ws->Y + off, rows, pc);

            /// Update trailing green panel in [Algorithm 1, Wang et al. 2025]
            // W = Y·T
            cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, pc, &one, ws->Y + off, lda, ws->Tri, pc, &zero,
                         ws->W + off, lda);

            // P = A[off]·W
            cublas::symm(ws, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, rows, pc, &one, A + tr, lda, ws->W + off, lda,
                         &zero, ws->Z + zc, lda);

            // Subtract previous panel contribution from P (w = block_cols):
            //   P -= Y[j+b:n, i:i+w]·(Z[j+b:n, 0:w]ᵀ·W) + Z[j+b:n, 0:w]·(Y[j+b:n, i:i+w]ᵀ·W)
            if (block_cols > 0)
            {
                // D = Z[j+b:n, 0:w]ᵀ·W
                cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, block_cols, pc, rows, &one, ws->Z + (j + pc), lda,
                             ws->W + off, lda, &zero, ws->Dwk, block_cols);
                // P -= Y[j+b:n, i:i+w]·D
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, block_cols, &neg1, ws->Y + i * lda + (j + pc), lda,
                             ws->Dwk, block_cols, &one, ws->Z + zc, lda);
                // D = Y[j+b:n, i:i+w]ᵀ·W
                cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, block_cols, pc, rows, &one, ws->Y + i * lda + (j + pc), lda,
                             ws->W + off, lda, &zero, ws->Dwk, block_cols);
                // P -= Z[j+b:n, 0:w]·D
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, block_cols, &neg1, ws->Z + (j + pc), lda, ws->Dwk,
                             block_cols, &one, ws->Z + zc, lda);
            }

            // C = Wᵀ·P
            cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, pc, pc, rows, &one, ws->W + off, lda, ws->Z + zc, lda, &zero,
                         ws->Dwk, pc);

            // Z = P - 0.5·W·C
            cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, pc, &neg_half, ws->Y + off, lda, ws->Dwk, pc, &one,
                         ws->Z + zc, lda);

            block_cols += pc;

            // green update if next panel is within this block
            if (j + pc < i + kc)
            {
                // clamp to the matrix edge; a full-pc walk can run past column n-1
                const int nc = std::min(pc, rows);
                // A[j+b:n, j+b:j+2b] -= Z[j+b:n, 0:w]·Y[j+b:j+2b, i:i+w]ᵀ
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_T, rows, nc, block_cols, &neg1, ws->Z + (j + pc), lda,
                             ws->Y + i * lda + (j + pc), lda, &one, A + tr, lda);
                // A[j+b:n, j+b:j+2b] -= Y[j+b:n, i:i+w]·Z[j+b:j+2b, 0:w]ᵀ
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_T, rows, nc, block_cols, &neg1, ws->Y + i * lda + (j + pc), lda,
                             ws->Z + (j + pc), lda, &one, A + tr, lda);
            }
        }

        // Update trailing green block in [Algorithm 1, Wang et al. 2025]:
        //   A[i+k:n, i+k:n] -= Y[i+k:n, i:i+w]·Z[i+k:n, 0:w]ᵀ + Z[i+k:n, 0:w]·Y[i+k:n, i:i+w]ᵀ
        cublas::syr2k(ws, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n - (i + kc), block_cols, &neg1,
                      ws->Y + i * lda + (i + kc), lda, ws->Z + (i + kc), lda, &one, A + (i + kc) * lda + (i + kc), lda);
    }

    // Pack the lower band of the reduced matrix into packed B
    dbbr_pack(ws, A, B, n, b);
}

} // namespace kernels
} // namespace ase
