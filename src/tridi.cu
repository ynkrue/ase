/**
 * @file   tridi.cu
 * @brief  Tridiagonal divide-and-conquer eigensolver driver — GPU-native.
 *
 * Cuppen D&C with LAPACK dstedc's structure, but with the eigenvector matrix Q
 * device-resident in `evec` for the whole solve. Per merge, only O(m) vectors
 * (z, d, permutations) cross PCIe; every super-linear cost runs on the GPU:
 *
 *   host   deflation index logic     line-by-line port of dlaed2 (O(m), branchy)
 *   device Givens rotations + column gather (deflation bookkeeping on Q)
 *   device secular solve + eigenvector assembly  see secular.cu
 *   cuBLAS rank-1 update GEMMs       Q ← Q2·S, exploiting the type-1/2/3 zero block
 *                                    structure (two block GEMMs, LAPACK-style)
 *
 * Leaves (≤ DC_LEAF) are solved on the host with LAPACK *stedc, OpenMP-parallel
 * across the independent leaves, and uploaded once (O(n·DC_LEAF) data total).
 *
 * Device scratch: Q2 gather lives in ws->M, the k×k secular delta matrix in
 * ws->Sdc, and the assembled S matrix in the caller-provided scratch (the user's
 * A buffer — already consumed by DBBR). Small per-merge staging in ws->dc_*.
 *
 * Host scratch is handle-owned (ws->h_*): the merge loop performs no allocation.
 * Two stream synchronisations remain per merge, both genuine data dependencies of
 * the host-side dlaed2 port — it needs the coupling vector z before it can decide
 * the deflation, and the secular roots before it can build the next permutation.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-07
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

// =============================================================================
// LAPACK (host) f77 entry points — leaf solver + merge-sort permutation only.
// =============================================================================
extern "C"
{
    void dstedc_(const char*, const int*, double*, double*, double*, const int*, double*, const int*, int*, const int*,
                 int*);
    void dlamrg_(const int*, const int*, const double*, const int*, const int*, int*);
}

// =============================================================================
// Device kernels
// =============================================================================
namespace
{

/// z = [row (n1−1) of the Q1 block | row n1 of the Q2 block] (dlaed1's coupling vector).
__global__ void dc_extract_z_kernel(const double* __restrict__ Q, long ldq, int m, int n1, double* __restrict__ z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    z[i] = (i < n1) ? Q[(n1 - 1) + (size_t)i * ldq] : Q[n1 + (size_t)i * ldq];
}

/// Apply the deflation Givens rotations in order (thread = row, so chaining is exact).
__global__ void dc_rot_cols_kernel(double* Q, long ldq, int m, int nrot, const int* __restrict__ ij,
                                   const double* __restrict__ cs)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    for (int t = 0; t < nrot; ++t)
    {
        double*      xp = Q + r + (size_t)ij[2 * t] * ldq;
        double*      yp = Q + r + (size_t)ij[2 * t + 1] * ldq;
        const double c = cs[2 * t], s = cs[2 * t + 1];
        const double x = *xp, y = *yp;
        *xp = c * x + s * y;
        *yp = c * y - s * x;
    }
}

/// out(:,p) = Q(:, map[p]) — column gather.
__global__ void dc_gather_cols_kernel(const double* __restrict__ Q, long ldq, double* __restrict__ out, long ldo,
                                      int rows, int cols, const int* __restrict__ map)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int p = blockIdx.y; p < cols; p += gridDim.y)
        out[r + (size_t)p * ldo] = Q[r + (size_t)map[p] * ldq];
}

/// Q[0] = 1 — identity eigenvector for a 1×1 irreducible block.
__global__ void dc_set_one_kernel(double* q) { *q = 1.0; }

} // namespace

namespace ase
{
namespace kernels
{

namespace
{

// -----------------------------------------------------------------------------
// Host-side helpers
// -----------------------------------------------------------------------------
void lap_stedc(const char* compz, int n, double* d, double* e, double* Z, int ldz, double* work, int lwork, int* iwork,
               int liwork, int* info)
{ dstedc_(compz, &n, d, e, Z, &ldz, work, &lwork, iwork, &liwork, info); }

void lap_lamrg(int n1, int n2, const double* a, int d1, int d2, int* index) { dlamrg_(&n1, &n2, a, &d1, &d2, index); }

[[noreturn]] void fail(const char* what, int info)
{
    fprintf(stderr, "tridi_dc: %s failed, info = %d\n", what, info);
    exit(EXIT_FAILURE);
}

double tridi_maxnorm(int n, const double* d, const double* e)
{
    double m = 0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(d[i]));
    for (int i = 0; i < n - 1; ++i)
        m = std::max(m, std::abs(e[i]));
    return m;
}

/// Deflation bookkeeping produced by host_laed2 for the device pipeline.
struct DcMerge
{
    int k       = 0;            ///< non-deflated roots
    int nrot    = 0;            ///< Givens rotations recorded in ij/cs
    int ctot[4] = {0, 0, 0, 0}; ///< column-type counts (1-, mixed, 2-block, deflated)
};

// -----------------------------------------------------------------------------
// host_laed2 — deflation for one rank-1 merge; index logic only (port of
// Reference-LAPACK dlaed2.f with the Q-column DROT/DCOPY operations replaced by
// a rotation list (ij/cs) and gather permutation (indx) executed on the device).
//
// In:  m, n1, rho (by ref: scaled to |2ρ|), d (natural order), z (coupling
//      vector, natural order), indxq (per-half sort permutations, 0-based).
// Out: dlamda/w (k, dlamda order), indx (m, type-grouped gather order),
//      ixc (k, S-row gather), ij/cs (rotations), d updated (deflated pairs mixed,
//      deflated values moved to the tail d[k..m) in LAPACK's descending order).
// iwk: 3m ints of scratch.
// -----------------------------------------------------------------------------
DcMerge host_laed2(int m, int n1, double& rho, double* d, double* z, int* indxq, double* dlamda, double* w, int* indx,
                   int* ixc, int* ij, double* cs, int* iwk)
{
    const double eps   = lapack_eps();
    int *        indxm = iwk, *indxp = iwk + m, *coltyp = iwk + 2 * m;
    DcMerge      r;

    if (rho < 0)
        for (int i = n1; i < m; ++i)
            z[i] = -z[i];
    const double t = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < m; ++i)
        z[i] *= t;
    rho = std::abs(2 * rho);

    for (int i = n1; i < m; ++i)
        indxq[i] += n1;
    for (int i = 0; i < m; ++i)
        dlamda[i] = d[indxq[i]];
    // merge the two sorted halves (dlamrg with strides 1,1)
    {
        int ia = 0, ib = n1, p = 0;
        while (ia < n1 && ib < m)
            indxm[p++] = (dlamda[ia] <= dlamda[ib]) ? indxq[ia++] : indxq[ib++];
        while (ia < n1)
            indxm[p++] = indxq[ia++];
        while (ib < m)
            indxm[p++] = indxq[ib++];
    }

    int imax = 0, jmax = 0;
    for (int i = 1; i < m; ++i)
        if (std::abs(z[i]) > std::abs(z[imax])) imax = i;
    for (int i = 1; i < m; ++i)
        if (std::abs(d[i]) > std::abs(d[jmax])) jmax = i;
    const double tol = 8 * eps * std::max(std::abs(d[jmax]), std::abs(z[imax]));

    // rank-1 modifier negligible: columns only need sorting
    if (rho * std::abs(z[imax]) <= tol)
    {
        r.k = 0;
        for (int j = 0; j < m; ++j)
        {
            indx[j]   = indxm[j];
            dlamda[j] = d[indxm[j]];
        }
        for (int j = 0; j < m; ++j)
            d[j] = dlamda[j];
        return r;
    }

    for (int i = 0; i < n1; ++i)
        coltyp[i] = 0;
    for (int i = n1; i < m; ++i)
        coltyp[i] = 2;

    int k = 0, k2 = m;
    int pj = 0, j = 0;
    // skip leading small-z entries (all deflate)
    for (; j < m; ++j)
    {
        const int nj = indxm[j];
        if (rho * std::abs(z[nj]) <= tol)
        {
            --k2;
            coltyp[nj] = 3;
            indxp[k2]  = nj;
        }
        else
        {
            pj = nj;
            ++j;
            break;
        }
    }
    for (; j < m; ++j)
    {
        const int nj = indxm[j];
        if (rho * std::abs(z[nj]) <= tol)
        {
            --k2;
            coltyp[nj] = 3;
            indxp[k2]  = nj;
        }
        else
        {
            double       s_ = z[pj], c_ = z[nj];
            const double tau  = std::hypot(c_, s_);
            const double tdif = d[nj] - d[pj];
            c_ /= tau;
            s_ = -s_ / tau;
            if (std::abs(tdif * c_ * s_) <= tol)
            {
                // deflate pj by rotating (pj, nj)
                z[nj] = tau;
                z[pj] = 0.0;
                if (coltyp[nj] != coltyp[pj]) coltyp[nj] = 1;
                coltyp[pj]         = 3;
                ij[2 * r.nrot]     = pj;
                ij[2 * r.nrot + 1] = nj;
                cs[2 * r.nrot]     = c_;
                cs[2 * r.nrot + 1] = s_;
                ++r.nrot;
                const double tt = d[pj] * c_ * c_ + d[nj] * s_ * s_;
                d[nj]           = d[pj] * s_ * s_ + d[nj] * c_ * c_;
                d[pj]           = tt;
                --k2;
                // insert pj into the (descending) deflated tail
                int i1 = 1;
                for (;;)
                {
                    if (k2 + i1 < m && d[pj] < d[indxp[k2 + i1]])
                    {
                        indxp[k2 + i1 - 1] = indxp[k2 + i1];
                        indxp[k2 + i1]     = pj;
                        ++i1;
                    }
                    else
                    {
                        indxp[k2 + i1 - 1] = pj;
                        break;
                    }
                }
                pj = nj;
            }
            else
            {
                dlamda[k] = d[pj];
                w[k]      = z[pj];
                indxp[k]  = pj;
                ++k;
                pj = nj;
            }
        }
    }
    dlamda[k] = d[pj];
    w[k]      = z[pj];
    indxp[k]  = pj;
    ++k;

    for (int q = 0; q < m; ++q)
        r.ctot[coltyp[q]]++;
    k   = m - r.ctot[3];
    r.k = k;

    int psm[4] = {0, r.ctot[0], r.ctot[0] + r.ctot[1], r.ctot[0] + r.ctot[1] + r.ctot[2]};
    for (int q = 0; q < m; ++q)
    {
        const int js  = indxp[q];
        const int ct  = coltyp[js];
        indx[psm[ct]] = js;
        ixc[psm[ct]]  = q;
        psm[ct]++;
    }

    // deflated eigenvalues into the tail of d (z is dead, reuse as scratch)
    for (int p = k; p < m; ++p)
        z[p] = d[indx[p]];
    for (int p = k; p < m; ++p)
        d[p] = z[p];
    return r;
}

// -----------------------------------------------------------------------------
// merge_gpu — one rank-1 merge (dlaed1): deflate on the host, everything O(m²)
// on the device. Qb is the m×m diagonal block of Q; S is ≥ m·m device scratch.
// -----------------------------------------------------------------------------
void merge_gpu(AseHandle* ws, double* Qb, int ldq, int m, int n1, double rho, double* d, int* indxq, double* S,
               int* iwk)
{
    cudaStream_t st = ws->stream;
    const size_t es = sizeof(double);

    // 1. coupling vector z → host
    dc_extract_z_kernel<<<div_up(m, 256), 256, 0, st>>>(Qb, (long)ldq, m, n1, ws->dc_z);
    CUDA_CHECK(cudaMemcpyAsync(ws->h_z, ws->dc_z, m * es, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    // 2. deflation (index logic only)
    const DcMerge r  = host_laed2(m, n1, rho, d, ws->h_z, indxq, ws->h_dlamda, ws->h_w, ws->h_indx, ws->h_ixc, ws->h_ij,
                                  ws->h_cs, iwk);
    const int     k  = r.k;
    double*       Q2 = ws->M;
    const int     ldq2 = m;
    const dim3    gcols(div_up(m, 256), std::min(m, 4096));

    CUDA_CHECK(cudaMemcpyAsync(ws->dc_indx, ws->h_indx, m * sizeof(int), cudaMemcpyHostToDevice, st));

    // rank-1 modifier negligible: sort columns, done
    if (k == 0)
    {
        dc_gather_cols_kernel<<<gcols, 256, 0, st>>>(Qb, (long)ldq, Q2, (long)ldq2, m, m, ws->dc_indx);
        CUDA_CHECK(
            cudaMemcpy2DAsync(Qb, (size_t)ldq * es, Q2, (size_t)ldq2 * es, m * es, m, cudaMemcpyDeviceToDevice, st));
        for (int i = 0; i < m; ++i)
            indxq[i] = i;
        return;
    }

    // 3. deflation Givens rotations on Q columns
    if (r.nrot)
    {
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_ij, ws->h_ij, 2 * r.nrot * sizeof(int), cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_cs, ws->h_cs, 2 * r.nrot * es, cudaMemcpyHostToDevice, st));
        dc_rot_cols_kernel<<<div_up(m, 256), 256, 0, st>>>(Qb, (long)ldq, m, r.nrot, ws->dc_ij, ws->dc_cs);
    }

    // 4. Q2 ← Q columns in type-grouped order (zero blocks ride along)
    dc_gather_cols_kernel<<<gcols, 256, 0, st>>>(Qb, (long)ldq, Q2, (long)ldq2, m, m, ws->dc_indx);

    // 5. secular solve + eigenvector assembly, all device-resident
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_dlamda, ws->h_dlamda, k * es, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_w, ws->h_w, k * es, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_ixc, ws->h_ixc, k * sizeof(int), cudaMemcpyHostToDevice, st));
    kernels::dc_secular_solve(ws, k, rho, S);

    // 6. eigenvector update: two block GEMMs straight into Q columns [0, k)
    const int    n2  = m - n1;
    const int    n12 = r.ctot[0] + r.ctot[1];
    const int    n23 = r.ctot[1] + r.ctot[2];
    const double one = 1.0, zero = 0.0;
    if (n23)
        cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, n2, k, n23, &one, Q2 + n1 + (size_t)r.ctot[0] * ldq2, ldq2,
                     S + r.ctot[0], k, &zero, Qb + n1, ldq);
    else
        CUDA_CHECK(cudaMemset2DAsync(Qb + n1, (size_t)ldq * es, 0, n2 * es, k, st));
    if (n12)
        cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, n1, k, n12, &one, Q2, ldq2, S, k, &zero, Qb, ldq);
    else
        CUDA_CHECK(cudaMemset2DAsync(Qb, (size_t)ldq * es, 0, n1 * es, k, st));
    // deflated columns back into Q(:, k..m)
    if (k < m)
        CUDA_CHECK(cudaMemcpy2DAsync(Qb + (size_t)k * ldq, (size_t)ldq * es, Q2 + (size_t)k * ldq2, (size_t)ldq2 * es,
                                     m * es, m - k, cudaMemcpyDeviceToDevice, st));

    // 7. eigenvalues → host, next-level permutation
    CUDA_CHECK(cudaMemcpyAsync(ws->h_lam, ws->dc_lam, k * es, cudaMemcpyDeviceToHost, st));
    int hinfo = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hinfo, ws->dc_info, sizeof(int), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (hinfo != 0) fail("laed4 (device)", hinfo);
    for (int i = 0; i < k; ++i)
        d[i] = ws->h_lam[i];

    lap_lamrg(k, m - k, d, 1, -1, indxq); // 1-based output
    for (int i = 0; i < m; ++i)
        indxq[i] -= 1;
}

/// Solve the bottom-level sub-problems on the host (*stedc, OpenMP across leaves)
/// and upload the block-diagonal eigenvector blocks onto the zeroed Q.
void solve_leaves(AseHandle* ws, int subpbs, const int* part, double* d, double* e, double* Qb, int ldq, int* indxq)
{
    size_t* qoff = ws->h_qoff;
    qoff[0]      = 0;
    for (int i = 0; i < subpbs; ++i)
    {
        const int ms = (i == 0) ? part[0] : part[i] - part[i - 1];
        qoff[i + 1]  = qoff[i] + (size_t)ms * ms;
    }
    double* hQ = ws->h_leafQ;

    // Each leaf owns a disjoint slice of the workspaces, addressed from the partition
    // rather than from a thread id: leaf i needs 1+4·ms+ms² doubles and 3+5·ms ints, and
    // laying them out in leaf order makes consecutive slices exactly abut.
    int nfail = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : nfail)
    for (int i = 0; i < subpbs; ++i)
    {
        const int sm     = (i == 0) ? 0 : part[i - 1];
        const int ms     = (i == 0) ? part[0] : part[i] - part[i - 1];
        const int lwork  = 1 + 4 * ms + ms * ms;
        const int liwork = 3 + 5 * ms;
        double*   work   = ws->h_leafwork + qoff[i] + 4 * (size_t)sm + i;
        int*      iwork  = ws->h_leafiwork + 5 * (size_t)sm + 3 * i;
        int       info   = 0;
        lap_stedc("I", ms, &d[sm], &e[sm], &hQ[qoff[i]], ms, work, lwork, iwork, liwork, &info);
        if (info != 0) ++nfail;
        for (int q = 0; q < ms; ++q)
            indxq[sm + q] = q;
    }
    if (nfail) fail("stedc (leaf)", nfail);

    for (int i = 0; i < subpbs; ++i)
    {
        const int sm = (i == 0) ? 0 : part[i - 1];
        const int ms = (i == 0) ? part[0] : part[i] - part[i - 1];
        CUDA_CHECK(cudaMemcpy2DAsync(Qb + sm + (size_t)sm * ldq, (size_t)ldq * sizeof(double), &hQ[qoff[i]],
                                     (size_t)ms * sizeof(double), ms * sizeof(double), ms, cudaMemcpyHostToDevice,
                                     ws->stream));
    }
}

// -----------------------------------------------------------------------------
// laex0_gpu — D&C over one irreducible block (dlaed0): partition, host leaves,
// bottom-up GPU merges, final permutation.
// -----------------------------------------------------------------------------
void laex0_gpu(AseHandle* ws, int bn, double* d, double* e, double* Qb, int ldq, double* S, int* indxq, int* iwk,
               int* part)
{
    part[0]    = bn;
    int subpbs = 1;
    while (part[subpbs - 1] > DC_LEAF)
    {
        for (int j = subpbs; j > 0; --j)
        {
            part[2 * j - 1] = (part[j - 1] + 1) / 2;
            part[2 * j - 2] = part[j - 1] / 2;
        }
        subpbs *= 2;
    }
    for (int j = 1; j < subpbs; ++j)
        part[j] += part[j - 1];

    // rank-1 split corrections on the diagonal
    for (int i = 0; i < subpbs - 1; ++i)
    {
        const int sm = part[i];
        d[sm - 1] -= std::abs(e[sm - 1]);
        d[sm] -= std::abs(e[sm - 1]);
    }

    solve_leaves(ws, subpbs, part, d, e, Qb, ldq, indxq);

    while (subpbs > 1)
    {
        for (int i = 0; i < subpbs - 1; i += 2)
        {
            int submat, matsiz, msd2;
            if (i == 0)
            {
                submat = 0;
                matsiz = part[1];
                msd2   = part[0];
            }
            else
            {
                submat = part[i - 1];
                matsiz = part[i + 1] - part[i - 1];
                msd2   = matsiz / 2;
            }
            merge_gpu(ws, Qb + submat + (size_t)submat * ldq, ldq, matsiz, msd2, e[submat + msd2 - 1], &d[submat],
                      &indxq[submat], S, iwk);
            part[i / 2] = part[i + 1];
        }
        subpbs /= 2;
    }

    // final ascending order: permute d on the host, gather Q columns on the device
    {
        double* dt = ws->h_dlamda; // dead between merges — reuse as reorder scratch
        for (int i = 0; i < bn; ++i)
            dt[i] = d[indxq[i]];
        for (int i = 0; i < bn; ++i)
            d[i] = dt[i];
        std::copy(indxq, indxq + bn, ws->h_indx);
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_indx, ws->h_indx, bn * sizeof(int), cudaMemcpyHostToDevice, ws->stream));
        const dim3 g(div_up(bn, 256), std::min(bn, 4096));
        dc_gather_cols_kernel<<<g, 256, 0, ws->stream>>>(Qb, (long)ldq, ws->M, (long)bn, bn, bn, ws->dc_indx);
        CUDA_CHECK(cudaMemcpy2DAsync(Qb, (size_t)ldq * sizeof(double), ws->M, (size_t)bn * sizeof(double),
                                     bn * sizeof(double), bn, cudaMemcpyDeviceToDevice, ws->stream));
    }
}

// -----------------------------------------------------------------------------
// stedx — top level (dstedc): split at negligible off-diagonals, scale each
// irreducible block to unit norm, solve (host ≤ DC_LEAF, GPU D&C above), and
// restore global ascending order if the matrix split.
// -----------------------------------------------------------------------------
void stedx(AseHandle* ws, int n, double* d, double* e, double* Q, int ldq, double* S)
{
    CUDA_CHECK(cudaMemsetAsync(Q, 0, (size_t)ldq * n * sizeof(double), ws->stream));

    int*         indxq = ws->h_indxq;
    const double eps   = lapack_eps();

    int nblocks = 0;
    int start   = 0;
    while (start < n)
    {
        int end = start + 1;
        for (; end < n; ++end)
        {
            // dstedc's splitting test, kept in its two-factor form: |a·b| can overflow or
            // flush to zero where sqrt|a|·sqrt|b| is exact.
            const double tiny = eps * std::sqrt(std::abs(d[end - 1])) * std::sqrt(std::abs(d[end]));
            if (std::abs(e[end - 1]) <= tiny) break;
        }
        const int m = end - start;
        ++nblocks;
        double* Qb = Q + start + (size_t)start * ldq;

        if (m == 1)
        {
            dc_set_one_kernel<<<1, 1, 0, ws->stream>>>(Qb);
            start = end;
            continue;
        }

        const double bn = tridi_maxnorm(m, &d[start], &e[start]);
        if (bn != 0.0)
        {
            for (int i = 0; i < m; ++i)
                d[start + i] /= bn;
            for (int i = 0; i < m - 1; ++i)
                e[start + i] /= bn;

            if (m > DC_LEAF)
            {
                laex0_gpu(ws, m, &d[start], &e[start], Qb, ldq, S, &indxq[start], ws->h_iwk, ws->h_part);
            }
            else
            {
                // single block, so it takes the first slice of the leaf workspaces
                int info = 0;
                lap_stedc("I", m, &d[start], &e[start], ws->h_leafQ, m, ws->h_leafwork, 1 + 4 * m + m * m,
                          ws->h_leafiwork, 3 + 5 * m, &info);
                if (info != 0) fail("stedc (block)", info);
                CUDA_CHECK(cudaMemcpy2DAsync(Qb, (size_t)ldq * sizeof(double), ws->h_leafQ, (size_t)m * sizeof(double),
                                             m * sizeof(double), m, cudaMemcpyHostToDevice, ws->stream));
                // h_leafQ is shared with the next block's solve, and consecutive small blocks
                // reach this point with no intervening merge to order the copy against.
                CUDA_CHECK(cudaStreamSynchronize(ws->stream));
            }

            for (int i = 0; i < m; ++i)
                d[start + i] *= bn;
        }
        start = end;
    }

    // matrix split into independent blocks: global sort of (d, Q columns)
    if (nblocks > 1)
    {
        int*    perm = ws->h_indx;   // sorted in place, then uploaded directly as the gather map
        double* dt   = ws->h_dlamda; // dead once the merges are done — reuse as reorder scratch
        std::iota(perm, perm + n, 0);
        std::stable_sort(perm, perm + n, [&](int a, int b) { return d[a] < d[b]; });
        for (int i = 0; i < n; ++i)
            dt[i] = d[perm[i]];
        std::copy(dt, dt + n, d);
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_indx, perm, n * sizeof(int), cudaMemcpyHostToDevice, ws->stream));
        const dim3 g(div_up(n, 256), std::min(n, 4096));
        dc_gather_cols_kernel<<<g, 256, 0, ws->stream>>>(Q, (long)ldq, ws->M, (long)n, n, n, ws->dc_indx);
        CUDA_CHECK(cudaMemcpy2DAsync(Q, (size_t)ldq * sizeof(double), ws->M, (size_t)n * sizeof(double),
                                     n * sizeof(double), n, cudaMemcpyDeviceToDevice, ws->stream));
    }
}

} // namespace

// =============================================================================
// Public entry: GPU tridiagonal D&C.
// =============================================================================
void tridi_dc(AseHandle* ws, double* d, double* e, double* eval, double* evec, double* scratch)
{
    const int n = ws->n;

    // tridiagonal → host (the only O(n) host state). The host D&C driver reads these
    // immediately, so this sync is a real data dependency, not a lifetime guard.
    ws->h_e[n - 1] = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(ws->h_d, d, n * sizeof(double), cudaMemcpyDeviceToHost, ws->stream));
    if (n > 1) CUDA_CHECK(cudaMemcpyAsync(ws->h_e, e, (n - 1) * sizeof(double), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    stedx(ws, n, ws->h_d, ws->h_e, evec, n, scratch);

    // h_d is handle-owned and the caller's next stream op is ordered after this copy, so
    // there is nothing to wait for here.
    CUDA_CHECK(cudaMemcpyAsync(eval, ws->h_d, n * sizeof(double), cudaMemcpyHostToDevice, ws->stream));
}

} // namespace kernels
} // namespace ase
