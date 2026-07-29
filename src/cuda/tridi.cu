/**
 * @file   tridi.cu
 * @brief  Tridiagonal divide-and-conquer eigensolver — GPU-native.
 *
 * Cuppen D&C with LAPACK dstedc's structure, but with the eigenvector matrix Q
 * device-resident in `evec` for the whole solve. Per merge, only O(m) vectors
 * (z, d, permutations) cross PCIe; every super-linear cost runs on the GPU:
 *
 *   host   deflation index logic     line-by-line port of dlaed2 (O(m), branchy)
 *   device Givens rotations + column gather (deflation bookkeeping on Q)
 *   device secular roots             dlaed4/dlaed5/dlaed6 ported to a __device__
 *                                    function, one thread per root
 *   device Gu-Eisenstat weight fixup + eigenvector assembly (dlaed3's O(k²) part)
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
 * @author  Yannik Rüfenacht
 * @date    2026-07
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

// =============================================================================
// LAPACK (host) f77 entry points — leaf solver + merge-sort permutation only.
// =============================================================================
extern "C" {
void dstedc_(const char *, const int *, double *, double *, double *, const int *, double *,
             const int *, int *, const int *, int *);
void sstedc_(const char *, const int *, float *, float *, float *, const int *, float *,
             const int *, int *, const int *, int *);
void dlamrg_(const int *, const int *, const double *, const int *, const int *, int *);
void slamrg_(const int *, const int *, const float *, const int *, const int *, int *);
}

// =============================================================================
// Device kernels
// =============================================================================
namespace {

/// Leaf size: sub-problems of this size or smaller are solved by host *stedc.
constexpr int DC_LEAF = 512;

/// LAPACK DLAMCH('Epsilon') = half the C epsilon (IEEE round-to-nearest ulp/2).
template <typename T> __host__ __device__ __forceinline__ T dc_eps() {
    return std::numeric_limits<T>::epsilon() / T(2);
}

// -----------------------------------------------------------------------------
// dlaed6 port — one root of a 3-pole secular equation (Gragg-Thornton-Warner).
// Direct translation of Reference-LAPACK dlaed6.f; d/z are 3-element arrays.
// -----------------------------------------------------------------------------
template <typename T>
__device__ void dev_laed6(int kniter, bool orgati, T rho, const T *d, const T *z, T finit, T &tau,
                          int &info) {
    constexpr int MAXIT = 40;
    const T zero = 0, one = 1, two = 2, four = 4, eight = 8;
    info = 0;

    T lbd, ubd;
    if (orgati) {
        lbd = d[1], ubd = d[2];
    } else {
        lbd = d[0], ubd = d[1];
    }
    if (finit < zero)
        lbd = zero;
    else
        ubd = zero;

    tau = zero;
    if (kniter == 2) {
        T temp, a, b, c;
        if (orgati) {
            temp = (d[2] - d[1]) / two;
            c = rho + z[0] / ((d[0] - d[1]) - temp);
            a = c * (d[1] + d[2]) + z[1] + z[2];
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1];
        } else {
            temp = (d[0] - d[1]) / two;
            c = rho + z[2] / ((d[2] - d[1]) - temp);
            a = c * (d[0] + d[1]) + z[0] + z[1];
            b = c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0];
        }
        temp = max(max(fabs(a), fabs(b)), fabs(c));
        a /= temp;
        b /= temp;
        c /= temp;
        if (c == zero)
            tau = b / a;
        else if (a <= zero)
            tau = (a - sqrt(fabs(a * a - four * b * c))) / (two * c);
        else
            tau = two * b / (a + sqrt(fabs(a * a - four * b * c)));
        if (tau < lbd || tau > ubd) tau = (lbd + ubd) / two;
        if (d[0] == tau || d[1] == tau || d[2] == tau) {
            tau = zero;
        } else {
            temp = finit + tau * z[0] / (d[0] * (d[0] - tau)) + tau * z[1] / (d[1] * (d[1] - tau)) +
                   tau * z[2] / (d[2] * (d[2] - tau));
            if (temp <= zero)
                lbd = tau;
            else
                ubd = tau;
            if (fabs(finit) <= fabs(temp)) tau = zero;
        }
    }

    const T eps = dc_eps<T>();
    // radix^int(log(safmin)/log(radix)/3): 2^-340 (double), 2^-42 (float)
    const T small1 = std::is_same<T, double>::value ? (T)ldexp(1.0, -340) : (T)ldexp(1.0f, -42);
    const T sminv1 = one / small1;
    const T small2 = small1 * small1;
    const T sminv2 = sminv1 * sminv1;

    T temp =
        orgati ? min(fabs(d[1] - tau), fabs(d[2] - tau)) : min(fabs(d[0] - tau), fabs(d[1] - tau));
    bool scale = false;
    T sclinv = one;
    T dscale[3], zscale[3];
    if (temp <= small1) {
        scale = true;
        T sclfac;
        if (temp <= small2) {
            sclfac = sminv2, sclinv = small2;
        } else {
            sclfac = sminv1, sclinv = small1;
        }
        for (int i = 0; i < 3; ++i) {
            dscale[i] = d[i] * sclfac;
            zscale[i] = z[i] * sclfac;
        }
        tau *= sclfac;
        lbd *= sclfac;
        ubd *= sclfac;
    } else {
        for (int i = 0; i < 3; ++i) {
            dscale[i] = d[i];
            zscale[i] = z[i];
        }
    }

    T fc = zero, df = zero, ddf = zero;
    for (int i = 0; i < 3; ++i) {
        T t1 = one / (dscale[i] - tau);
        T t2 = zscale[i] * t1;
        fc += t2 / dscale[i];
        df += t2 * t1;
        ddf += t2 * t1 * t1;
    }
    T f = finit + tau * fc;

    if (fabs(f) > zero) {
        if (f <= zero)
            lbd = tau;
        else
            ubd = tau;
        bool converged = false;
        for (int niter = 2; niter <= MAXIT; ++niter) {
            T temp1, temp2;
            if (orgati) {
                temp1 = dscale[1] - tau, temp2 = dscale[2] - tau;
            } else {
                temp1 = dscale[0] - tau, temp2 = dscale[1] - tau;
            }
            T a = (temp1 + temp2) * f - temp1 * temp2 * df;
            T b = temp1 * temp2 * f;
            T c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
            T tm = max(max(fabs(a), fabs(b)), fabs(c));
            a /= tm;
            b /= tm;
            c /= tm;
            T eta;
            if (c == zero)
                eta = b / a;
            else if (a <= zero)
                eta = (a - sqrt(fabs(a * a - four * b * c))) / (two * c);
            else
                eta = two * b / (a + sqrt(fabs(a * a - four * b * c)));
            if (f * eta >= zero) eta = -f / df;

            tau += eta;
            if (tau < lbd || tau > ubd) tau = (lbd + ubd) / two;

            fc = zero;
            T erretm = zero;
            df = zero;
            ddf = zero;
            bool hitpole = false;
            for (int i = 0; i < 3; ++i) {
                if ((dscale[i] - tau) != zero) {
                    T t1 = one / (dscale[i] - tau);
                    T t2 = zscale[i] * t1;
                    T t4 = t2 / dscale[i];
                    fc += t4;
                    erretm += fabs(t4);
                    df += t2 * t1;
                    ddf += t2 * t1 * t1;
                } else {
                    hitpole = true;
                }
            }
            if (hitpole) {
                converged = true;
                break;
            }
            f = finit + tau * fc;
            erretm = eight * (fabs(finit) + fabs(tau) * erretm) + fabs(tau) * df;
            if (fabs(f) <= four * eps * erretm || (ubd - lbd) <= four * eps * fabs(tau)) {
                converged = true;
                break;
            }
            if (f <= zero)
                lbd = tau;
            else
                ubd = tau;
        }
        if (!converged) info = 1;
    }
    if (scale) tau *= sclinv;
}

// -----------------------------------------------------------------------------
// dlaed5 port — k=2 secular equation; delta receives the *normalized eigenvector*.
// -----------------------------------------------------------------------------
template <typename T>
__device__ void dev_laed5(int i, const T *d, const T *z, T *delta, T rho, T &dlam) {
    const T zero = 0, one = 1, two = 2, four = 4;
    const T del = d[1] - d[0];
    if (i == 1) {
        T w = one + two * rho * (z[1] * z[1] - z[0] * z[0]) / del;
        if (w > zero) {
            T b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            T c = rho * z[0] * z[0] * del;
            T tau = two * c / (b + sqrt(fabs(b * b - four * c)));
            dlam = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        } else {
            T b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            T c = rho * z[1] * z[1] * del;
            T tau;
            if (b > zero)
                tau = -two * c / (b + sqrt(b * b + four * c));
            else
                tau = (b - sqrt(b * b + four * c)) / two;
            dlam = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }
    } else {
        T b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        T c = rho * z[1] * z[1] * del;
        T tau;
        if (b > zero)
            tau = (b + sqrt(b * b + four * c)) / two;
        else
            tau = two * c / (-b + sqrt(b * b + four * c));
        dlam = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
    }
    T temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
    delta[0] /= temp;
    delta[1] /= temp;
}

// -----------------------------------------------------------------------------
// dlaed4 port — root i (1-based) of the k-pole secular equation.
//
// Differs from the Fortran in one respect: the DELTA array is not materialised —
// the root is carried in the origin-shift representation λ = d[org] + tau, and
// every delta value the iteration needs is recomputed on the fly as
// (d[j] − d[org]) − tau (same accuracy, no O(k) array updates per iteration).
// D and Z are 1-based (caller passes ptr-1).
// -----------------------------------------------------------------------------
template <typename T>
__device__ void dev_laed4(int n, int i, const T *__restrict__ D, const T *__restrict__ Z, T rho,
                          T &tau_out, int &org_out, T &dlam, int &info) {
    constexpr int MAXIT = 30;
    const T zero = 0, one = 1, two = 2, three = 3, four = 4, eight = 8, ten = 10;
    info = 0;
    const T eps = dc_eps<T>();
    const T rhoinv = one / rho;

    if (i == n) {
        // ---- the case i = n -------------------------------------------------
        const int ii = n - 1;
        const T midpt = rho / two;

        T psi = zero;
        for (int j = 1; j <= n - 2; ++j)
            psi += Z[j] * Z[j] / ((D[j] - D[n]) - midpt);
        T c = rhoinv + psi;
        T w = c + Z[ii] * Z[ii] / ((D[ii] - D[n]) - midpt) + Z[n] * Z[n] / (-midpt);

        T tau, dltlb, dltub;
        if (w <= zero) {
            T temp = Z[n - 1] * Z[n - 1] / (D[n] - D[n - 1] + rho) + Z[n] * Z[n] / rho;
            if (c <= temp) {
                tau = rho;
            } else {
                T del = D[n] - D[n - 1];
                T a = -c * del + Z[n - 1] * Z[n - 1] + Z[n] * Z[n];
                T b = Z[n] * Z[n] * del;
                if (a < zero)
                    tau = two * b / (sqrt(a * a + four * b * c) - a);
                else
                    tau = (a + sqrt(a * a + four * b * c)) / (two * c);
            }
            dltlb = midpt;
            dltub = rho;
        } else {
            T del = D[n] - D[n - 1];
            T a = -c * del + Z[n - 1] * Z[n - 1] + Z[n] * Z[n];
            T b = Z[n] * Z[n] * del;
            if (a < zero)
                tau = two * b / (sqrt(a * a + four * b * c) - a);
            else
                tau = (a + sqrt(a * a + four * b * c)) / (two * c);
            dltlb = zero;
            dltub = midpt;
        }

        T dpsi, phi, dphi, erretm;
        auto evalw = [&](T tau_) {
            dpsi = zero;
            psi = zero;
            erretm = zero;
            for (int j = 1; j <= ii; ++j) {
                T temp = Z[j] / ((D[j] - D[n]) - tau_);
                psi += Z[j] * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = fabs(erretm);
            T temp = Z[n] / (-tau_);
            phi = Z[n] * temp;
            dphi = temp * temp;
            erretm = eight * (-phi - psi) + erretm - phi + rhoinv + fabs(tau_) * (dpsi + dphi);
        };
        evalw(tau);
        w = rhoinv + phi + psi;

        if (fabs(w) <= eps * erretm) {
            dlam = D[i] + tau;
            tau_out = tau;
            org_out = n;
            return;
        }
        if (w <= zero)
            dltlb = max(dltlb, tau);
        else
            dltub = min(dltub, tau);

        // first step (has the C<0 / geometric-mean handling)
        {
            T dn1 = (D[n - 1] - D[n]) - tau;
            T dn = -tau;
            T c2 = w - dn1 * dpsi - dn * dphi;
            T a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
            T b = dn1 * dn * w;
            if (c2 < zero) c2 = fabs(c2);
            T eta;
            if (c2 == zero)
                eta = -w / (dpsi + dphi);
            else if (a >= zero)
                eta = (a + sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            else
                eta = two * b / (a - sqrt(fabs(a * a - four * b * c2)));
            if (w * eta > zero) eta = -w / (dpsi + dphi);
            T temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                T eta1 = -w / (dpsi + dphi);
                temp = tau + eta1;
                T eta2 = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
                if (dltlb <= temp && temp <= dltub)
                    eta = copysign(one, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
                else
                    eta = eta2;
            }
            tau += eta;
        }
        evalw(tau);
        w = rhoinv + phi + psi;

        for (int niter = 3; niter <= MAXIT; ++niter) {
            if (fabs(w) <= eps * erretm) {
                dlam = D[i] + tau;
                tau_out = tau;
                org_out = n;
                return;
            }
            if (w <= zero)
                dltlb = max(dltlb, tau);
            else
                dltub = min(dltub, tau);

            T dn1 = (D[n - 1] - D[n]) - tau;
            T dn = -tau;
            T c2 = w - dn1 * dpsi - dn * dphi;
            T a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
            T b = dn1 * dn * w;
            T eta;
            if (a >= zero)
                eta = (a + sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            else
                eta = two * b / (a - sqrt(fabs(a * a - four * b * c2)));
            if (w * eta > zero) eta = -w / (dpsi + dphi);
            T temp = tau + eta;
            if (temp > dltub || temp < dltlb)
                eta = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
            tau += eta;
            evalw(tau);
            w = rhoinv + phi + psi;
        }
        info = 1;
        dlam = D[i] + tau;
        tau_out = tau;
        org_out = n;
        return;
    }

    // ---- the case i < n -----------------------------------------------------
    const int ip1 = i + 1;
    const T del = D[ip1] - D[i];
    const T midpt = del / two;

    T psi = zero;
    for (int j = 1; j <= i - 1; ++j)
        psi += Z[j] * Z[j] / ((D[j] - D[i]) - midpt);
    T phi = zero;
    for (int j = n; j >= i + 2; --j)
        phi += Z[j] * Z[j] / ((D[j] - D[i]) - midpt);
    T c = rhoinv + psi + phi;
    T w = c + Z[i] * Z[i] / (-midpt) + Z[ip1] * Z[ip1] / ((D[ip1] - D[i]) - midpt);

    bool orgati;
    T tau, dltlb, dltub;
    if (w > zero) {
        orgati = true;
        T a = c * del + Z[i] * Z[i] + Z[ip1] * Z[ip1];
        T b = Z[i] * Z[i] * del;
        if (a > zero)
            tau = two * b / (a + sqrt(fabs(a * a - four * b * c)));
        else
            tau = (a - sqrt(fabs(a * a - four * b * c))) / (two * c);
        dltlb = zero;
        dltub = midpt;
    } else {
        orgati = false;
        T a = c * del - Z[i] * Z[i] - Z[ip1] * Z[ip1];
        T b = Z[ip1] * Z[ip1] * del;
        if (a < zero)
            tau = two * b / (a - sqrt(fabs(a * a + four * b * c)));
        else
            tau = -(a + sqrt(fabs(a * a + four * b * c))) / (two * c);
        dltlb = -midpt;
        dltub = zero;
    }
    const int ii = orgati ? i : ip1;
    const int iim1 = ii - 1, iip1 = ii + 1;

    // delta_j at the current shift (origin d[ii])
    auto dj = [&](int j_, T tau_) { return (D[j_] - D[ii]) - tau_; };

    T dpsi, dphi, erretm;
    auto evalw2 = [&](T tau_) {
        dpsi = zero;
        psi = zero;
        erretm = zero;
        for (int j = 1; j <= iim1; ++j) {
            T t = Z[j] / dj(j, tau_);
            psi += Z[j] * t;
            dpsi += t * t;
            erretm += psi;
        }
        erretm = fabs(erretm);
        dphi = zero;
        phi = zero;
        for (int j = n; j >= iip1; --j) {
            T t = Z[j] / dj(j, tau_);
            phi += Z[j] * t;
            dphi += t * t;
            erretm += phi;
        }
    };
    evalw2(tau);
    w = rhoinv + phi + psi;

    bool swtch3 = orgati ? (w < zero) : (w > zero);
    if (ii == 1 || ii == n) swtch3 = false;

    T temp = Z[ii] / dj(ii, tau);
    T dw = dpsi + dphi + temp * temp;
    temp = Z[ii] * temp;
    w += temp;
    erretm = eight * (phi - psi) + erretm + two * rhoinv + three * fabs(temp) + fabs(tau) * dw;

    if (fabs(w) <= eps * erretm) {
        dlam = D[ii] + tau;
        tau_out = tau;
        org_out = ii;
        return;
    }
    if (w <= zero)
        dltlb = max(dltlb, tau);
    else
        dltub = min(dltub, tau);

    // first step (kniter = 2)
    T eta;
    {
        if (!swtch3) {
            T c2, a, b;
            if (orgati) {
                T t1 = Z[i] / dj(i, tau);
                c2 = w - dj(ip1, tau) * dw - (D[i] - D[ip1]) * t1 * t1;
            } else {
                T t1 = Z[ip1] / dj(ip1, tau);
                c2 = w - dj(i, tau) * dw - (D[ip1] - D[i]) * t1 * t1;
            }
            a = (dj(i, tau) + dj(ip1, tau)) * w - dj(i, tau) * dj(ip1, tau) * dw;
            b = dj(i, tau) * dj(ip1, tau) * w;
            if (c2 == zero) {
                if (a == zero) {
                    if (orgati)
                        a = Z[i] * Z[i] + dj(ip1, tau) * dj(ip1, tau) * (dpsi + dphi);
                    else
                        a = Z[ip1] * Z[ip1] + dj(i, tau) * dj(i, tau) * (dpsi + dphi);
                }
                eta = b / a;
            } else if (a <= zero) {
                eta = (a - sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            } else {
                eta = two * b / (a + sqrt(fabs(a * a - four * b * c2)));
            }
        } else {
            // three-pole interpolation
            T tmp = rhoinv + psi + phi;
            T c2, zz0, zz2;
            if (orgati) {
                T t1 = Z[iim1] / dj(iim1, tau);
                t1 = t1 * t1;
                c2 = tmp - dj(iip1, tau) * (dpsi + dphi) - (D[iim1] - D[iip1]) * t1;
                zz0 = Z[iim1] * Z[iim1];
                zz2 = dj(iip1, tau) * dj(iip1, tau) * ((dpsi - t1) + dphi);
            } else {
                T t1 = Z[iip1] / dj(iip1, tau);
                t1 = t1 * t1;
                c2 = tmp - dj(iim1, tau) * (dpsi + dphi) - (D[iip1] - D[iim1]) * t1;
                zz0 = dj(iim1, tau) * dj(iim1, tau) * (dpsi + (dphi - t1));
                zz2 = Z[iip1] * Z[iip1];
            }
            T dd3[3] = {dj(iim1, tau), dj(ii, tau), dj(iip1, tau)};
            T zz3[3] = {zz0, Z[ii] * Z[ii], zz2};
            int info6 = 0;
            dev_laed6(2, orgati, c2, dd3, zz3, w, eta, info6);
            if (info6 != 0) {
                info = info6;
                dlam = D[ii] + tau;
                tau_out = tau;
                org_out = ii;
                return;
            }
        }
        if (w * eta >= zero) eta = -w / dw;
        T tmp2 = tau + eta;
        if (tmp2 > dltub || tmp2 < dltlb)
            eta = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
    }

    T prew = w;
    tau += eta;
    evalw2(tau);
    temp = Z[ii] / dj(ii, tau);
    dw = dpsi + dphi + temp * temp;
    temp = Z[ii] * temp;
    w = rhoinv + phi + psi + temp;
    erretm = eight * (phi - psi) + erretm + two * rhoinv + three * fabs(temp) + fabs(tau) * dw;

    bool swtch = orgati ? (-w > fabs(prew) / ten) : (w > fabs(prew) / ten);

    for (int niter = 3; niter <= MAXIT; ++niter) {
        if (fabs(w) <= eps * erretm) {
            dlam = D[ii] + tau;
            tau_out = tau;
            org_out = ii;
            return;
        }
        if (w <= zero)
            dltlb = max(dltlb, tau);
        else
            dltub = min(dltub, tau);

        if (!swtch3) {
            T c2, a, b;
            if (!swtch) {
                if (orgati) {
                    T t1 = Z[i] / dj(i, tau);
                    c2 = w - dj(ip1, tau) * dw - (D[i] - D[ip1]) * t1 * t1;
                } else {
                    T t1 = Z[ip1] / dj(ip1, tau);
                    c2 = w - dj(i, tau) * dw - (D[ip1] - D[i]) * t1 * t1;
                }
            } else {
                temp = Z[ii] / dj(ii, tau);
                if (orgati)
                    dpsi += temp * temp;
                else
                    dphi += temp * temp;
                c2 = w - dj(i, tau) * dpsi - dj(ip1, tau) * dphi;
            }
            a = (dj(i, tau) + dj(ip1, tau)) * w - dj(i, tau) * dj(ip1, tau) * dw;
            b = dj(i, tau) * dj(ip1, tau) * w;
            if (c2 == zero) {
                if (a == zero) {
                    if (!swtch) {
                        if (orgati)
                            a = Z[i] * Z[i] + dj(ip1, tau) * dj(ip1, tau) * (dpsi + dphi);
                        else
                            a = Z[ip1] * Z[ip1] + dj(i, tau) * dj(i, tau) * (dpsi + dphi);
                    } else {
                        a = dj(i, tau) * dj(i, tau) * dpsi + dj(ip1, tau) * dj(ip1, tau) * dphi;
                    }
                }
                eta = b / a;
            } else if (a <= zero) {
                eta = (a - sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            } else {
                eta = two * b / (a + sqrt(fabs(a * a - four * b * c2)));
            }
        } else {
            // three-pole interpolation
            T tmp = rhoinv + psi + phi;
            T c2, zz0, zz2;
            if (swtch) {
                c2 = tmp - dj(iim1, tau) * dpsi - dj(iip1, tau) * dphi;
                zz0 = dj(iim1, tau) * dj(iim1, tau) * dpsi;
                zz2 = dj(iip1, tau) * dj(iip1, tau) * dphi;
            } else {
                if (orgati) {
                    T t1 = Z[iim1] / dj(iim1, tau);
                    t1 = t1 * t1;
                    c2 = tmp - dj(iip1, tau) * (dpsi + dphi) - (D[iim1] - D[iip1]) * t1;
                    zz0 = Z[iim1] * Z[iim1];
                    zz2 = dj(iip1, tau) * dj(iip1, tau) * ((dpsi - t1) + dphi);
                } else {
                    T t1 = Z[iip1] / dj(iip1, tau);
                    t1 = t1 * t1;
                    c2 = tmp - dj(iim1, tau) * (dpsi + dphi) - (D[iip1] - D[iim1]) * t1;
                    zz0 = dj(iim1, tau) * dj(iim1, tau) * (dpsi + (dphi - t1));
                    zz2 = Z[iip1] * Z[iip1];
                }
            }
            T dd3[3] = {dj(iim1, tau), dj(ii, tau), dj(iip1, tau)};
            T zz3[3] = {zz0, Z[ii] * Z[ii], zz2};
            int info6 = 0;
            dev_laed6(niter, orgati, c2, dd3, zz3, w, eta, info6);
            if (info6 != 0) {
                info = info6;
                dlam = D[ii] + tau;
                tau_out = tau;
                org_out = ii;
                return;
            }
        }
        if (w * eta >= zero) eta = -w / dw;
        T tmp2 = tau + eta;
        if (tmp2 > dltub || tmp2 < dltlb) {
            T eta1 = -w / dw;
            tmp2 = tau + eta1;
            T eta2 = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
            if (dltlb <= tmp2 && tmp2 <= dltub)
                eta = copysign(one, eta1) * sqrt(fabs(eta1)) * sqrt(fabs(eta2));
            else
                eta = eta2;
        }
        tau += eta;
        prew = w;
        evalw2(tau);
        temp = Z[ii] / dj(ii, tau);
        dw = dpsi + dphi + temp * temp;
        temp = Z[ii] * temp;
        w = rhoinv + phi + psi + temp;
        erretm = eight * (phi - psi) + erretm + two * rhoinv + three * fabs(temp) + fabs(tau) * dw;
        if (w * prew > zero && fabs(w) > fabs(prew) / ten) swtch = !swtch;
    }
    info = 1;
    dlam = D[ii] + tau;
    tau_out = tau;
    org_out = ii;
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

/**
 * @brief Secular equation solver: one thread per root j.
 *
 * k ≥ 3: outputs the origin-shift representation λ_j = dlamda[org_j] + tau_j
 * (delta matrix filled separately by dc_fill_delta_kernel).
 * k ≤ 2: dlaed5 writes the normalized eigenvector straight into delta column j.
 */
template <typename T>
__global__ void dc_laed4_kernel(int k, const T *__restrict__ dlamda, const T *__restrict__ w, T rho,
                                T *delta, T *dtau, int *dorg, T *dlam, int *info) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    if (k == 1) {
        dlam[0] = dlamda[0] + rho * w[0] * w[0];
        delta[0] = T(1);
        return;
    }
    if (k == 2) {
        T dl;
        dev_laed5(j + 1, dlamda, w, delta + (size_t)j * k, rho, dl);
        dlam[j] = dl;
        return;
    }
    T tau, dl;
    int org, linfo = 0;
    dev_laed4<T>(k, j + 1, dlamda - 1, w - 1, rho, tau, org, dl, linfo);
    dtau[j] = tau;
    dorg[j] = org - 1;
    dlam[j] = dl;
    if (linfo != 0) atomicExch(info, linfo);
}

/// delta(i,j) = (dlamda[i] − dlamda[org_j]) − tau_j, column-major k×k (coalesced in i).
template <typename T>
__global__ void dc_fill_delta_kernel(int k, const T *__restrict__ dlamda,
                                     const T *__restrict__ dtau, const int *__restrict__ dorg,
                                     T *__restrict__ delta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const T di = dlamda[i];
    for (int j = blockIdx.y; j < k; j += gridDim.y)
        delta[i + (size_t)j * k] = (di - dlamda[dorg[j]]) - dtau[j];
}

/// Gu-Eisenstat stabilized weights (dlaed3): w̃_i = sign(w_i)·√(−Π_j δ_ij/(λ_i−λ_j)).
template <typename T>
__global__ void dc_weights_kernel(int k, const T *__restrict__ dlamda, const T *__restrict__ delta,
                                  const T *__restrict__ w_in, T *__restrict__ w_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const T di = dlamda[i];
    T p = delta[i + (size_t)i * k];
    for (int j = 0; j < i; ++j)
        p *= delta[i + (size_t)j * k] / (di - dlamda[j]);
    for (int j = i + 1; j < k; ++j)
        p *= delta[i + (size_t)j * k] / (di - dlamda[j]);
    w_out[i] = copysign(sqrt(-p), w_in[i]);
}

/**
 * @brief Assemble the rank-1 update eigenvector matrix S (dlaed3's O(k²) part).
 *
 * One block per column j. normalize ≠ 0 (k ≥ 3): s_i = w̃_i/δ_ij, column
 * 2-normalized, rows gathered by ixc so S lines up with Q2's type-grouped column
 * order. normalize = 0 (k ≤ 2): delta already holds the normalized eigenvector,
 * gather rows only.
 */
template <typename T, int NT>
__global__ void dc_form_s_kernel(int k, const T *__restrict__ delta, const T *__restrict__ w,
                                 const int *__restrict__ ixc, T *__restrict__ S, int normalize) {
    const int j = blockIdx.x;
    const T *dcol = delta + (size_t)j * k;
    T *scol = S + (size_t)j * k;
    if (!normalize) {
        for (int p = threadIdx.x; p < k; p += NT)
            scol[p] = dcol[ixc[p]];
        return;
    }
    __shared__ T smem[NT];
    __shared__ T snorm;
    T loc = T(0);
    for (int i = threadIdx.x; i < k; i += NT) {
        T s = w[i] / dcol[i];
        loc += s * s;
    }
    T tot = block_reduce_sum<T, NT>(loc, smem);
    if (threadIdx.x == 0) snorm = sqrt(tot);
    __syncthreads();
    const T t = snorm;
    for (int p = threadIdx.x; p < k; p += NT) {
        const int ii = ixc[p];
        scol[p] = (w[ii] / dcol[ii]) / t;
    }
}

/// z = [row (n1−1) of the Q1 block | row n1 of the Q2 block] (dlaed1's coupling vector).
template <typename T>
__global__ void dc_extract_z_kernel(const T *__restrict__ Q, long ldq, int m, int n1,
                                    T *__restrict__ z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    z[i] = (i < n1) ? Q[(n1 - 1) + (size_t)i * ldq] : Q[n1 + (size_t)i * ldq];
}

/// Apply the deflation Givens rotations in order (thread = row, so chaining is exact).
template <typename T>
__global__ void dc_rot_cols_kernel(T *Q, long ldq, int m, int nrot, const int *__restrict__ ij,
                                   const T *__restrict__ cs) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    for (int t = 0; t < nrot; ++t) {
        T *xp = Q + r + (size_t)ij[2 * t] * ldq;
        T *yp = Q + r + (size_t)ij[2 * t + 1] * ldq;
        const T c = cs[2 * t], s = cs[2 * t + 1];
        const T x = *xp, y = *yp;
        *xp = c * x + s * y;
        *yp = c * y - s * x;
    }
}

/// out(:,p) = Q(:, map[p]) — column gather.
template <typename T>
__global__ void dc_gather_cols_kernel(const T *__restrict__ Q, long ldq, T *__restrict__ out,
                                      long ldo, int rows, int cols, const int *__restrict__ map) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int p = blockIdx.y; p < cols; p += gridDim.y)
        out[r + (size_t)p * ldo] = Q[r + (size_t)map[p] * ldq];
}

/// Q[0] = 1 — identity eigenvector for a 1×1 irreducible block.
template <typename T> __global__ void dc_set_one_kernel(T *q) {
    *q = T(1);
}

} // namespace

namespace cuev {
namespace kernels {

namespace {

// -----------------------------------------------------------------------------
// Host-side helpers
// -----------------------------------------------------------------------------
template <typename T>
void lap_stedc(const char *compz, int n, T *d, T *e, T *Z, int ldz, T *work, int lwork, int *iwork,
               int liwork, int *info) {
    if constexpr (std::is_same_v<T, float>)
        sstedc_(compz, &n, d, e, Z, &ldz, work, &lwork, iwork, &liwork, info);
    else
        dstedc_(compz, &n, d, e, Z, &ldz, work, &lwork, iwork, &liwork, info);
}

template <typename T> void lap_lamrg(int n1, int n2, const T *a, int d1, int d2, int *index) {
    if constexpr (std::is_same_v<T, float>)
        slamrg_(&n1, &n2, a, &d1, &d2, index);
    else
        dlamrg_(&n1, &n2, a, &d1, &d2, index);
}

[[noreturn]] void fail(const char *what, int info) {
    fprintf(stderr, "tridi_dc: %s failed, info = %d\n", what, info);
    exit(EXIT_FAILURE);
}

template <typename T> T tridi_maxnorm(int n, const T *d, const T *e) {
    T m = 0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(d[i]));
    for (int i = 0; i < n - 1; ++i)
        m = std::max(m, std::abs(e[i]));
    return m;
}

/// Deflation bookkeeping produced by host_laed2 for the device pipeline.
struct DcMerge {
    int k = 0;                  ///< non-deflated roots
    int nrot = 0;               ///< Givens rotations recorded in ij/cs
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
template <typename T>
DcMerge host_laed2(int m, int n1, T &rho, T *d, T *z, int *indxq, T *dlamda, T *w, int *indx,
                   int *ixc, int *ij, T *cs, int *iwk) {
    const T eps = dc_eps<T>();
    int *indxm = iwk, *indxp = iwk + m, *coltyp = iwk + 2 * m;
    DcMerge r;

    if (rho < 0)
        for (int i = n1; i < m; ++i)
            z[i] = -z[i];
    const T t = T(1) / std::sqrt(T(2));
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
    const T tol = 8 * eps * std::max(std::abs(d[jmax]), std::abs(z[imax]));

    // rank-1 modifier negligible: columns only need sorting
    if (rho * std::abs(z[imax]) <= tol) {
        r.k = 0;
        for (int j = 0; j < m; ++j) {
            indx[j] = indxm[j];
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
    for (; j < m; ++j) {
        const int nj = indxm[j];
        if (rho * std::abs(z[nj]) <= tol) {
            --k2;
            coltyp[nj] = 3;
            indxp[k2] = nj;
        } else {
            pj = nj;
            ++j;
            break;
        }
    }
    for (; j < m; ++j) {
        const int nj = indxm[j];
        if (rho * std::abs(z[nj]) <= tol) {
            --k2;
            coltyp[nj] = 3;
            indxp[k2] = nj;
        } else {
            T s_ = z[pj], c_ = z[nj];
            const T tau = std::hypot(c_, s_);
            const T tdif = d[nj] - d[pj];
            c_ /= tau;
            s_ = -s_ / tau;
            if (std::abs(tdif * c_ * s_) <= tol) {
                // deflate pj by rotating (pj, nj)
                z[nj] = tau;
                z[pj] = T(0);
                if (coltyp[nj] != coltyp[pj]) coltyp[nj] = 1;
                coltyp[pj] = 3;
                ij[2 * r.nrot] = pj;
                ij[2 * r.nrot + 1] = nj;
                cs[2 * r.nrot] = c_;
                cs[2 * r.nrot + 1] = s_;
                ++r.nrot;
                const T tt = d[pj] * c_ * c_ + d[nj] * s_ * s_;
                d[nj] = d[pj] * s_ * s_ + d[nj] * c_ * c_;
                d[pj] = tt;
                --k2;
                // insert pj into the (descending) deflated tail
                int i1 = 1;
                for (;;) {
                    if (k2 + i1 < m && d[pj] < d[indxp[k2 + i1]]) {
                        indxp[k2 + i1 - 1] = indxp[k2 + i1];
                        indxp[k2 + i1] = pj;
                        ++i1;
                    } else {
                        indxp[k2 + i1 - 1] = pj;
                        break;
                    }
                }
                pj = nj;
            } else {
                dlamda[k] = d[pj];
                w[k] = z[pj];
                indxp[k] = pj;
                ++k;
                pj = nj;
            }
        }
    }
    dlamda[k] = d[pj];
    w[k] = z[pj];
    indxp[k] = pj;
    ++k;

    for (int q = 0; q < m; ++q)
        r.ctot[coltyp[q]]++;
    k = m - r.ctot[3];
    r.k = k;

    int psm[4] = {0, r.ctot[0], r.ctot[0] + r.ctot[1], r.ctot[0] + r.ctot[1] + r.ctot[2]};
    for (int q = 0; q < m; ++q) {
        const int js = indxp[q];
        const int ct = coltyp[js];
        indx[psm[ct]] = js;
        ixc[psm[ct]] = q;
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
template <typename T>
void merge_gpu(SolverHandle<T> *ws, T *Qb, int ldq, int m, int n1, T rho, T *d, int *indxq, T *S,
               int *iwk) {
    cudaStream_t st = ws->stream;
    const size_t es = sizeof(T);

    // 1. coupling vector z → host
    dc_extract_z_kernel<<<div_up(m, 256), 256, 0, st>>>(Qb, (long)ldq, m, n1, ws->dc_z);
    CUDA_CHECK(cudaMemcpyAsync(ws->h_z, ws->dc_z, m * es, cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    // 2. deflation (index logic only)
    const DcMerge r = host_laed2<T>(m, n1, rho, d, ws->h_z, indxq, ws->h_dlamda, ws->h_w,
                                    ws->h_indx, ws->h_ixc, ws->h_ij, ws->h_cs, iwk);
    const int k = r.k;
    T *Q2 = ws->M;
    const int ldq2 = m;
    const dim3 gcols(div_up(m, 256), std::min(m, 4096));

    CUDA_CHECK(
        cudaMemcpyAsync(ws->dc_indx, ws->h_indx, m * sizeof(int), cudaMemcpyHostToDevice, st));

    // rank-1 modifier negligible: sort columns, done
    if (k == 0) {
        dc_gather_cols_kernel<<<gcols, 256, 0, st>>>(Qb, (long)ldq, Q2, (long)ldq2, m, m,
                                                     ws->dc_indx);
        CUDA_CHECK(cudaMemcpy2DAsync(Qb, (size_t)ldq * es, Q2, (size_t)ldq2 * es, m * es, m,
                                     cudaMemcpyDeviceToDevice, st));
        for (int i = 0; i < m; ++i)
            indxq[i] = i;
        return;
    }

    // 3. deflation Givens rotations on Q columns
    if (r.nrot) {
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_ij, ws->h_ij, 2 * r.nrot * sizeof(int),
                                   cudaMemcpyHostToDevice, st));
        CUDA_CHECK(
            cudaMemcpyAsync(ws->dc_cs, ws->h_cs, 2 * r.nrot * es, cudaMemcpyHostToDevice, st));
        dc_rot_cols_kernel<<<div_up(m, 256), 256, 0, st>>>(Qb, (long)ldq, m, r.nrot, ws->dc_ij,
                                                           ws->dc_cs);
    }

    // 4. Q2 ← Q columns in type-grouped order (zero blocks ride along)
    dc_gather_cols_kernel<<<gcols, 256, 0, st>>>(Qb, (long)ldq, Q2, (long)ldq2, m, m, ws->dc_indx);

    // 5. secular solve + eigenvector assembly, all device-resident
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_dlamda, ws->h_dlamda, k * es, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_w, ws->h_w, k * es, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ws->dc_ixc, ws->h_ixc, k * sizeof(int), cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemsetAsync(ws->dc_info, 0, sizeof(int), st));
    dc_laed4_kernel<<<div_up(k, 128), 128, 0, st>>>(
        k, ws->dc_dlamda, ws->dc_w, rho, ws->Sdc, ws->dc_tau, ws->dc_org, ws->dc_lam, ws->dc_info);
    if (k > 2) {
        const dim3 g(div_up(k, 256), std::min(k, 1024));
        dc_fill_delta_kernel<<<g, 256, 0, st>>>(k, ws->dc_dlamda, ws->dc_tau, ws->dc_org, ws->Sdc);
        dc_weights_kernel<<<div_up(k, 256), 256, 0, st>>>(k, ws->dc_dlamda, ws->Sdc, ws->dc_w,
                                                          ws->dc_wt);
    }
    dc_form_s_kernel<T, 256>
        <<<k, 256, 0, st>>>(k, ws->Sdc, ws->dc_wt, ws->dc_ixc, S, k > 2 ? 1 : 0);

    // 6. eigenvector update: two block GEMMs straight into Q columns [0, k)
    const int n2 = m - n1;
    const int n12 = r.ctot[0] + r.ctot[1];
    const int n23 = r.ctot[1] + r.ctot[2];
    const T one = T(1), zero = T(0);
    if (n23)
        cublas::gemm<T>(ws, CUBLAS_OP_N, CUBLAS_OP_N, n2, k, n23, &one,
                        Q2 + n1 + (size_t)r.ctot[0] * ldq2, ldq2, S + r.ctot[0], k, &zero, Qb + n1,
                        ldq);
    else
        CUDA_CHECK(cudaMemset2DAsync(Qb + n1, (size_t)ldq * es, 0, n2 * es, k, st));
    if (n12)
        cublas::gemm<T>(ws, CUBLAS_OP_N, CUBLAS_OP_N, n1, k, n12, &one, Q2, ldq2, S, k, &zero, Qb,
                        ldq);
    else
        CUDA_CHECK(cudaMemset2DAsync(Qb, (size_t)ldq * es, 0, n1 * es, k, st));
    // deflated columns back into Q(:, k..m)
    if (k < m)
        CUDA_CHECK(cudaMemcpy2DAsync(Qb + (size_t)k * ldq, (size_t)ldq * es, Q2 + (size_t)k * ldq2,
                                     (size_t)ldq2 * es, m * es, m - k, cudaMemcpyDeviceToDevice,
                                     st));

    // 7. eigenvalues → host, next-level permutation
    CUDA_CHECK(cudaMemcpyAsync(ws->h_lam, ws->dc_lam, k * es, cudaMemcpyDeviceToHost, st));
    int hinfo = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hinfo, ws->dc_info, sizeof(int), cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (hinfo != 0) fail("laed4 (device)", hinfo);
    for (int i = 0; i < k; ++i)
        d[i] = ws->h_lam[i];

    lap_lamrg<T>(k, m - k, d, 1, -1, indxq); // 1-based output
    for (int i = 0; i < m; ++i)
        indxq[i] -= 1;
}

/// Solve the bottom-level sub-problems on the host (*stedc, OpenMP across leaves)
/// and upload the block-diagonal eigenvector blocks onto the zeroed Q.
template <typename T>
void solve_leaves(SolverHandle<T> *ws, int subpbs, const int *part, T *d, T *e, T *Qb, int ldq,
                  int *indxq) {
    std::vector<size_t> qoff(subpbs + 1, 0);
    for (int i = 0; i < subpbs; ++i) {
        const int ms = (i == 0) ? part[0] : part[i] - part[i - 1];
        qoff[i + 1] = qoff[i] + (size_t)ms * ms;
    }
    std::vector<T> hQ(qoff[subpbs]);

    int nfail = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : nfail)
    for (int i = 0; i < subpbs; ++i) {
        const int sm = (i == 0) ? 0 : part[i - 1];
        const int ms = (i == 0) ? part[0] : part[i] - part[i - 1];
        const int lwork = 1 + 4 * ms + ms * ms;
        const int liwork = 3 + 5 * ms;
        std::vector<T> work(lwork);
        std::vector<int> iwork(liwork);
        int info = 0;
        lap_stedc<T>("I", ms, &d[sm], &e[sm], &hQ[qoff[i]], ms, work.data(), lwork, iwork.data(),
                     liwork, &info);
        if (info != 0) ++nfail;
        for (int q = 0; q < ms; ++q)
            indxq[sm + q] = q;
    }
    if (nfail) fail("stedc (leaf)", nfail);

    for (int i = 0; i < subpbs; ++i) {
        const int sm = (i == 0) ? 0 : part[i - 1];
        const int ms = (i == 0) ? part[0] : part[i] - part[i - 1];
        CUDA_CHECK(cudaMemcpy2DAsync(Qb + sm + (size_t)sm * ldq, (size_t)ldq * sizeof(T),
                                     &hQ[qoff[i]], (size_t)ms * sizeof(T), ms * sizeof(T), ms,
                                     cudaMemcpyHostToDevice, ws->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ws->stream)); // hQ freed on return
}

// -----------------------------------------------------------------------------
// laex0_gpu — D&C over one irreducible block (dlaed0): partition, host leaves,
// bottom-up GPU merges, final permutation.
// -----------------------------------------------------------------------------
template <typename T>
void laex0_gpu(SolverHandle<T> *ws, int bn, T *d, T *e, T *Qb, int ldq, T *S, int *indxq, int *iwk,
               int *part) {
    part[0] = bn;
    int subpbs = 1;
    while (part[subpbs - 1] > DC_LEAF) {
        for (int j = subpbs; j > 0; --j) {
            part[2 * j - 1] = (part[j - 1] + 1) / 2;
            part[2 * j - 2] = part[j - 1] / 2;
        }
        subpbs *= 2;
    }
    for (int j = 1; j < subpbs; ++j)
        part[j] += part[j - 1];

    // rank-1 split corrections on the diagonal
    for (int i = 0; i < subpbs - 1; ++i) {
        const int sm = part[i];
        d[sm - 1] -= std::abs(e[sm - 1]);
        d[sm] -= std::abs(e[sm - 1]);
    }

    solve_leaves(ws, subpbs, part, d, e, Qb, ldq, indxq);

    while (subpbs > 1) {
        for (int i = 0; i < subpbs - 1; i += 2) {
            int submat, matsiz, msd2;
            if (i == 0) {
                submat = 0;
                matsiz = part[1];
                msd2 = part[0];
            } else {
                submat = part[i - 1];
                matsiz = part[i + 1] - part[i - 1];
                msd2 = matsiz / 2;
            }
            merge_gpu(ws, Qb + submat + (size_t)submat * ldq, ldq, matsiz, msd2,
                      e[submat + msd2 - 1], &d[submat], &indxq[submat], S, iwk);
            part[i / 2] = part[i + 1];
        }
        subpbs /= 2;
    }

    // final ascending order: permute d on the host, gather Q columns on the device
    {
        T *dt = ws->h_dlamda; // dead between merges — reuse as reorder scratch
        for (int i = 0; i < bn; ++i)
            dt[i] = d[indxq[i]];
        for (int i = 0; i < bn; ++i)
            d[i] = dt[i];
        std::copy(indxq, indxq + bn, ws->h_indx);
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_indx, ws->h_indx, bn * sizeof(int),
                                   cudaMemcpyHostToDevice, ws->stream));
        const dim3 g(div_up(bn, 256), std::min(bn, 4096));
        dc_gather_cols_kernel<<<g, 256, 0, ws->stream>>>(Qb, (long)ldq, ws->M, (long)bn, bn, bn,
                                                         ws->dc_indx);
        CUDA_CHECK(cudaMemcpy2DAsync(Qb, (size_t)ldq * sizeof(T), ws->M, (size_t)bn * sizeof(T),
                                     bn * sizeof(T), bn, cudaMemcpyDeviceToDevice, ws->stream));
    }
}

// -----------------------------------------------------------------------------
// stedx — top level (dstedc): split at negligible off-diagonals, scale each
// irreducible block to unit norm, solve (host ≤ DC_LEAF, GPU D&C above), and
// restore global ascending order if the matrix split.
// -----------------------------------------------------------------------------
template <typename T> void stedx(SolverHandle<T> *ws, int n, T *d, T *e, T *Q, int ldq, T *S) {
    CUDA_CHECK(cudaMemsetAsync(Q, 0, (size_t)ldq * n * sizeof(T), ws->stream));

    std::vector<int> indxq(n), iwk(3 * n), part(2 * n + 2);
    const T eps = std::numeric_limits<T>::epsilon();

    int nblocks = 0;
    int start = 0;
    while (start < n) {
        int end = start + 1;
        for (; end < n; ++end) {
            const T tiny = eps * std::sqrt(std::abs(d[end - 1] * d[end]));
            if (std::abs(e[end - 1]) <= tiny) break;
        }
        const int m = end - start;
        ++nblocks;
        T *Qb = Q + start + (size_t)start * ldq;

        if (m == 1) {
            dc_set_one_kernel<<<1, 1, 0, ws->stream>>>(Qb);
            start = end;
            continue;
        }

        const T bn = tridi_maxnorm(m, &d[start], &e[start]);
        if (bn != T(0)) {
            for (int i = 0; i < m; ++i)
                d[start + i] /= bn;
            for (int i = 0; i < m - 1; ++i)
                e[start + i] /= bn;

            if (m > DC_LEAF) {
                laex0_gpu(ws, m, &d[start], &e[start], Qb, ldq, S, &indxq[start], iwk.data(),
                          part.data());
            } else {
                const int lwork = 1 + 4 * m + m * m;
                const int liwork = 3 + 5 * m;
                std::vector<T> hQ((size_t)m * m), work(lwork);
                std::vector<int> iwork(liwork);
                int info = 0;
                lap_stedc<T>("I", m, &d[start], &e[start], hQ.data(), m, work.data(), lwork,
                             iwork.data(), liwork, &info);
                if (info != 0) fail("stedc (block)", info);
                CUDA_CHECK(cudaMemcpy2DAsync(Qb, (size_t)ldq * sizeof(T), hQ.data(),
                                             (size_t)m * sizeof(T), m * sizeof(T), m,
                                             cudaMemcpyHostToDevice, ws->stream));
                CUDA_CHECK(cudaStreamSynchronize(ws->stream));
            }

            for (int i = 0; i < m; ++i)
                d[start + i] *= bn;
        }
        start = end;
    }

    // matrix split into independent blocks: global sort of (d, Q columns)
    if (nblocks > 1) {
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) { return d[a] < d[b]; });
        std::vector<T> dt(n);
        for (int i = 0; i < n; ++i)
            dt[i] = d[perm[i]];
        std::copy(dt.begin(), dt.end(), d);
        std::copy(perm.begin(), perm.end(), ws->h_indx);
        CUDA_CHECK(cudaMemcpyAsync(ws->dc_indx, ws->h_indx, n * sizeof(int), cudaMemcpyHostToDevice,
                                   ws->stream));
        const dim3 g(div_up(n, 256), std::min(n, 4096));
        dc_gather_cols_kernel<<<g, 256, 0, ws->stream>>>(Q, (long)ldq, ws->M, (long)n, n, n,
                                                         ws->dc_indx);
        CUDA_CHECK(cudaMemcpy2DAsync(Q, (size_t)ldq * sizeof(T), ws->M, (size_t)n * sizeof(T),
                                     n * sizeof(T), n, cudaMemcpyDeviceToDevice, ws->stream));
    }
}

} // namespace

// =============================================================================
// Public entry: GPU tridiagonal D&C.
// =============================================================================
template <typename T> void tridi_dc(SolverHandle<T> *ws, T *d, T *e, T *eval, T *evec, T *scratch) {
    const int n = ws->n;

    // tridiagonal → host (the only O(n) host state)
    std::vector<T> hd(n), he(n, T(0));
    CUDA_CHECK(cudaMemcpyAsync(hd.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, ws->stream));
    if (n > 1)
        CUDA_CHECK(
            cudaMemcpyAsync(he.data(), e, (n - 1) * sizeof(T), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    stedx<T>(ws, n, hd.data(), he.data(), evec, n, scratch);

    CUDA_CHECK(cudaMemcpyAsync(eval, hd.data(), n * sizeof(T), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T) template void tridi_dc<T>(SolverHandle<T> *, T *, T *, T *, T *, T *);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
