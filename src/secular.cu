/**
 * @file   secular.cu
 * @brief  Secular equation solver — GPU port of LAPACK dlaed4/dlaed5/dlaed6.
 *
 * Solves the rank-1 update secular equation for the D&C tridiagonal eigensolver
 * (tridi.cu): given the non-deflated eigenvalues dlamda and rank-1 weights w of a
 * merge, finds each root λ_j = dlamda[org_j] + tau_j (k ≥ 3, dlaed4) or the closed
 * form for k ≤ 2 (dlaed5), falling back to the 3-pole interpolation (dlaed6) inside
 * dlaed4's iteration. Gu-Eisenstat weight fixup and the O(k²) eigenvector assembly
 * (dlaed3's non-root part) round out the merge's device-resident work; only the O(m)
 * deflation index logic (dlaed2) stays on the host, in tridi.cu.
 *
 * One thread per root throughout; dc_secular_solve is the single entry point tridi.cu
 * calls per merge.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-07
 */

#include "common.h"
#include "handle.h"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// -----------------------------------------------------------------------------
// dlaed6 port — one root of a 3-pole secular equation (Gragg-Thornton-Warner).
// Direct translation of Reference-LAPACK dlaed6.f; d/z are 3-element arrays.
// -----------------------------------------------------------------------------
__device__ void dev_laed6(int kniter, bool orgati, double rho, const double *d, const double *z,
                          double finit, double &tau, int &info) {
    constexpr int MAXIT = 40;
    const double zero = 0, one = 1, two = 2, four = 4, eight = 8;
    info = 0;

    double lbd, ubd;
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
        double temp, a, b, c;
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

    const double eps = lapack_eps();
    // radix^int(log(safmin)/log(radix)/3): 2^-340
    const double small1 = ldexp(1.0, -340);
    const double sminv1 = one / small1;
    const double small2 = small1 * small1;
    const double sminv2 = sminv1 * sminv1;

    double temp =
        orgati ? min(fabs(d[1] - tau), fabs(d[2] - tau)) : min(fabs(d[0] - tau), fabs(d[1] - tau));
    bool scale = false;
    double sclinv = one;
    double dscale[3], zscale[3];
    if (temp <= small1) {
        scale = true;
        double sclfac;
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

    double fc = zero, df = zero, ddf = zero;
    for (int i = 0; i < 3; ++i) {
        double t1 = one / (dscale[i] - tau);
        double t2 = zscale[i] * t1;
        fc += t2 / dscale[i];
        df += t2 * t1;
        ddf += t2 * t1 * t1;
    }
    double f = finit + tau * fc;

    if (fabs(f) > zero) {
        if (f <= zero)
            lbd = tau;
        else
            ubd = tau;
        bool converged = false;
        for (int niter = 2; niter <= MAXIT; ++niter) {
            double temp1, temp2;
            if (orgati) {
                temp1 = dscale[1] - tau, temp2 = dscale[2] - tau;
            } else {
                temp1 = dscale[0] - tau, temp2 = dscale[1] - tau;
            }
            double a = (temp1 + temp2) * f - temp1 * temp2 * df;
            double b = temp1 * temp2 * f;
            double c = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
            double tm = max(max(fabs(a), fabs(b)), fabs(c));
            a /= tm;
            b /= tm;
            c /= tm;
            double eta;
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
            double erretm = zero;
            df = zero;
            ddf = zero;
            bool hitpole = false;
            for (int i = 0; i < 3; ++i) {
                if ((dscale[i] - tau) != zero) {
                    double t1 = one / (dscale[i] - tau);
                    double t2 = zscale[i] * t1;
                    double t4 = t2 / dscale[i];
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
__device__ void dev_laed5(int i, const double *d, const double *z, double *delta, double rho,
                          double &dlam) {
    const double zero = 0, one = 1, two = 2, four = 4;
    const double del = d[1] - d[0];
    if (i == 1) {
        double w = one + two * rho * (z[1] * z[1] - z[0] * z[0]) / del;
        if (w > zero) {
            double b = del + rho * (z[0] * z[0] + z[1] * z[1]);
            double c = rho * z[0] * z[0] * del;
            double tau = two * c / (b + sqrt(fabs(b * b - four * c)));
            dlam = d[0] + tau;
            delta[0] = -z[0] / tau;
            delta[1] = z[1] / (del - tau);
        } else {
            double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
            double c = rho * z[1] * z[1] * del;
            double tau;
            if (b > zero)
                tau = -two * c / (b + sqrt(b * b + four * c));
            else
                tau = (b - sqrt(b * b + four * c)) / two;
            dlam = d[1] + tau;
            delta[0] = -z[0] / (del + tau);
            delta[1] = -z[1] / tau;
        }
    } else {
        double b = -del + rho * (z[0] * z[0] + z[1] * z[1]);
        double c = rho * z[1] * z[1] * del;
        double tau;
        if (b > zero)
            tau = (b + sqrt(b * b + four * c)) / two;
        else
            tau = two * c / (-b + sqrt(b * b + four * c));
        dlam = d[1] + tau;
        delta[0] = -z[0] / (del + tau);
        delta[1] = -z[1] / tau;
    }
    double temp = sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
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
__device__ void dev_laed4(int n, int i, const double *__restrict__ D, const double *__restrict__ Z,
                          double rho, double &tau_out, int &org_out, double &dlam, int &info) {
    constexpr int MAXIT = 30;
    const double zero = 0, one = 1, two = 2, three = 3, four = 4, eight = 8, ten = 10;
    info = 0;
    const double eps = lapack_eps();
    const double rhoinv = one / rho;

    if (i == n) {
        // ---- the case i = n -------------------------------------------------
        const int ii = n - 1;
        const double midpt = rho / two;

        double psi = zero;
        for (int j = 1; j <= n - 2; ++j)
            psi += Z[j] * Z[j] / ((D[j] - D[n]) - midpt);
        double c = rhoinv + psi;
        double w = c + Z[ii] * Z[ii] / ((D[ii] - D[n]) - midpt) + Z[n] * Z[n] / (-midpt);

        double tau, dltlb, dltub;
        if (w <= zero) {
            double temp = Z[n - 1] * Z[n - 1] / (D[n] - D[n - 1] + rho) + Z[n] * Z[n] / rho;
            if (c <= temp) {
                tau = rho;
            } else {
                double del = D[n] - D[n - 1];
                double a = -c * del + Z[n - 1] * Z[n - 1] + Z[n] * Z[n];
                double b = Z[n] * Z[n] * del;
                if (a < zero)
                    tau = two * b / (sqrt(a * a + four * b * c) - a);
                else
                    tau = (a + sqrt(a * a + four * b * c)) / (two * c);
            }
            dltlb = midpt;
            dltub = rho;
        } else {
            double del = D[n] - D[n - 1];
            double a = -c * del + Z[n - 1] * Z[n - 1] + Z[n] * Z[n];
            double b = Z[n] * Z[n] * del;
            if (a < zero)
                tau = two * b / (sqrt(a * a + four * b * c) - a);
            else
                tau = (a + sqrt(a * a + four * b * c)) / (two * c);
            dltlb = zero;
            dltub = midpt;
        }

        double dpsi, phi, dphi, erretm;
        auto evalw = [&](double tau_) {
            dpsi = zero;
            psi = zero;
            erretm = zero;
            for (int j = 1; j <= ii; ++j) {
                double temp = Z[j] / ((D[j] - D[n]) - tau_);
                psi += Z[j] * temp;
                dpsi += temp * temp;
                erretm += psi;
            }
            erretm = fabs(erretm);
            double temp = Z[n] / (-tau_);
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
            double dn1 = (D[n - 1] - D[n]) - tau;
            double dn = -tau;
            double c2 = w - dn1 * dpsi - dn * dphi;
            double a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
            double b = dn1 * dn * w;
            if (c2 < zero) c2 = fabs(c2);
            double eta;
            if (c2 == zero)
                eta = -w / (dpsi + dphi);
            else if (a >= zero)
                eta = (a + sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            else
                eta = two * b / (a - sqrt(fabs(a * a - four * b * c2)));
            if (w * eta > zero) eta = -w / (dpsi + dphi);
            double temp = tau + eta;
            if (temp > dltub || temp < dltlb) {
                double eta1 = -w / (dpsi + dphi);
                temp = tau + eta1;
                double eta2 = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
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

            double dn1 = (D[n - 1] - D[n]) - tau;
            double dn = -tau;
            double c2 = w - dn1 * dpsi - dn * dphi;
            double a = (dn1 + dn) * w - dn1 * dn * (dpsi + dphi);
            double b = dn1 * dn * w;
            double eta;
            if (a >= zero)
                eta = (a + sqrt(fabs(a * a - four * b * c2))) / (two * c2);
            else
                eta = two * b / (a - sqrt(fabs(a * a - four * b * c2)));
            if (w * eta > zero) eta = -w / (dpsi + dphi);
            double temp = tau + eta;
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
    const double del = D[ip1] - D[i];
    const double midpt = del / two;

    double psi = zero;
    for (int j = 1; j <= i - 1; ++j)
        psi += Z[j] * Z[j] / ((D[j] - D[i]) - midpt);
    double phi = zero;
    for (int j = n; j >= i + 2; --j)
        phi += Z[j] * Z[j] / ((D[j] - D[i]) - midpt);
    double c = rhoinv + psi + phi;
    double w = c + Z[i] * Z[i] / (-midpt) + Z[ip1] * Z[ip1] / ((D[ip1] - D[i]) - midpt);

    bool orgati;
    double tau, dltlb, dltub;
    if (w > zero) {
        orgati = true;
        double a = c * del + Z[i] * Z[i] + Z[ip1] * Z[ip1];
        double b = Z[i] * Z[i] * del;
        if (a > zero)
            tau = two * b / (a + sqrt(fabs(a * a - four * b * c)));
        else
            tau = (a - sqrt(fabs(a * a - four * b * c))) / (two * c);
        dltlb = zero;
        dltub = midpt;
    } else {
        orgati = false;
        double a = c * del - Z[i] * Z[i] - Z[ip1] * Z[ip1];
        double b = Z[ip1] * Z[ip1] * del;
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
    auto dj = [&](int j_, double tau_) { return (D[j_] - D[ii]) - tau_; };

    double dpsi, dphi, erretm;
    auto evalw2 = [&](double tau_) {
        dpsi = zero;
        psi = zero;
        erretm = zero;
        for (int j = 1; j <= iim1; ++j) {
            double t = Z[j] / dj(j, tau_);
            psi += Z[j] * t;
            dpsi += t * t;
            erretm += psi;
        }
        erretm = fabs(erretm);
        dphi = zero;
        phi = zero;
        for (int j = n; j >= iip1; --j) {
            double t = Z[j] / dj(j, tau_);
            phi += Z[j] * t;
            dphi += t * t;
            erretm += phi;
        }
    };
    evalw2(tau);
    w = rhoinv + phi + psi;

    bool swtch3 = orgati ? (w < zero) : (w > zero);
    if (ii == 1 || ii == n) swtch3 = false;

    double temp = Z[ii] / dj(ii, tau);
    double dw = dpsi + dphi + temp * temp;
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
    double eta;
    {
        if (!swtch3) {
            double c2, a, b;
            if (orgati) {
                double t1 = Z[i] / dj(i, tau);
                c2 = w - dj(ip1, tau) * dw - (D[i] - D[ip1]) * t1 * t1;
            } else {
                double t1 = Z[ip1] / dj(ip1, tau);
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
            double tmp = rhoinv + psi + phi;
            double c2, zz0, zz2;
            if (orgati) {
                double t1 = Z[iim1] / dj(iim1, tau);
                t1 = t1 * t1;
                c2 = tmp - dj(iip1, tau) * (dpsi + dphi) - (D[iim1] - D[iip1]) * t1;
                zz0 = Z[iim1] * Z[iim1];
                zz2 = dj(iip1, tau) * dj(iip1, tau) * ((dpsi - t1) + dphi);
            } else {
                double t1 = Z[iip1] / dj(iip1, tau);
                t1 = t1 * t1;
                c2 = tmp - dj(iim1, tau) * (dpsi + dphi) - (D[iip1] - D[iim1]) * t1;
                zz0 = dj(iim1, tau) * dj(iim1, tau) * (dpsi + (dphi - t1));
                zz2 = Z[iip1] * Z[iip1];
            }
            double dd3[3] = {dj(iim1, tau), dj(ii, tau), dj(iip1, tau)};
            double zz3[3] = {zz0, Z[ii] * Z[ii], zz2};
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
        double tmp2 = tau + eta;
        if (tmp2 > dltub || tmp2 < dltlb)
            eta = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
    }

    double prew = w;
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
            double c2, a, b;
            if (!swtch) {
                if (orgati) {
                    double t1 = Z[i] / dj(i, tau);
                    c2 = w - dj(ip1, tau) * dw - (D[i] - D[ip1]) * t1 * t1;
                } else {
                    double t1 = Z[ip1] / dj(ip1, tau);
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
            double tmp = rhoinv + psi + phi;
            double c2, zz0, zz2;
            if (swtch) {
                c2 = tmp - dj(iim1, tau) * dpsi - dj(iip1, tau) * dphi;
                zz0 = dj(iim1, tau) * dj(iim1, tau) * dpsi;
                zz2 = dj(iip1, tau) * dj(iip1, tau) * dphi;
            } else {
                if (orgati) {
                    double t1 = Z[iim1] / dj(iim1, tau);
                    t1 = t1 * t1;
                    c2 = tmp - dj(iip1, tau) * (dpsi + dphi) - (D[iim1] - D[iip1]) * t1;
                    zz0 = Z[iim1] * Z[iim1];
                    zz2 = dj(iip1, tau) * dj(iip1, tau) * ((dpsi - t1) + dphi);
                } else {
                    double t1 = Z[iip1] / dj(iip1, tau);
                    t1 = t1 * t1;
                    c2 = tmp - dj(iim1, tau) * (dpsi + dphi) - (D[iip1] - D[iim1]) * t1;
                    zz0 = dj(iim1, tau) * dj(iim1, tau) * (dpsi + (dphi - t1));
                    zz2 = Z[iip1] * Z[iip1];
                }
            }
            double dd3[3] = {dj(iim1, tau), dj(ii, tau), dj(iip1, tau)};
            double zz3[3] = {zz0, Z[ii] * Z[ii], zz2};
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
        double tmp2 = tau + eta;
        if (tmp2 > dltub || tmp2 < dltlb) {
            double eta1 = -w / dw;
            tmp2 = tau + eta1;
            double eta2 = (w < zero) ? (dltub - tau) / two : (dltlb - tau) / two;
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
__global__ void dc_laed4_kernel(int k, const double *__restrict__ dlamda,
                                const double *__restrict__ w, double rho, double *delta,
                                double *dtau, int *dorg, double *dlam, int *info) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    if (k == 1) {
        dlam[0] = dlamda[0] + rho * w[0] * w[0];
        delta[0] = 1.0;
        return;
    }
    if (k == 2) {
        double dl;
        dev_laed5(j + 1, dlamda, w, delta + (size_t)j * k, rho, dl);
        dlam[j] = dl;
        return;
    }
    double tau, dl;
    int org, linfo = 0;
    dev_laed4(k, j + 1, dlamda - 1, w - 1, rho, tau, org, dl, linfo);
    dtau[j] = tau;
    dorg[j] = org - 1;
    dlam[j] = dl;
    if (linfo != 0) atomicExch(info, linfo);
}

/// delta(i,j) = (dlamda[i] − dlamda[org_j]) − tau_j, column-major k×k (coalesced in i).
__global__ void dc_fill_delta_kernel(int k, const double *__restrict__ dlamda,
                                     const double *__restrict__ dtau, const int *__restrict__ dorg,
                                     double *__restrict__ delta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const double di = dlamda[i];
    for (int j = blockIdx.y; j < k; j += gridDim.y)
        delta[i + (size_t)j * k] = (di - dlamda[dorg[j]]) - dtau[j];
}

/// Gu-Eisenstat stabilized weights (dlaed3): w̃_i = sign(w_i)·√(−Π_j δ_ij/(λ_i−λ_j)).
__global__ void dc_weights_kernel(int k, const double *__restrict__ dlamda,
                                  const double *__restrict__ delta, const double *__restrict__ w_in,
                                  double *__restrict__ w_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const double di = dlamda[i];
    double p = delta[i + (size_t)i * k];
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
template <int NT>
__global__ void dc_form_s_kernel(int k, const double *__restrict__ delta,
                                 const double *__restrict__ w, const int *__restrict__ ixc,
                                 double *__restrict__ S, int normalize) {
    const int j = blockIdx.x;
    const double *dcol = delta + (size_t)j * k;
    double *scol = S + (size_t)j * k;
    if (!normalize) {
        for (int p = threadIdx.x; p < k; p += NT)
            scol[p] = dcol[ixc[p]];
        return;
    }
    __shared__ double smem[NT];
    __shared__ double snorm;
    double loc = 0.0;
    for (int i = threadIdx.x; i < k; i += NT) {
        double s = w[i] / dcol[i];
        loc += s * s;
    }
    double tot = block_reduce_sum<NT>(loc, smem);
    if (threadIdx.x == 0) snorm = sqrt(tot);
    __syncthreads();
    const double t = snorm;
    for (int p = threadIdx.x; p < k; p += NT) {
        const int ii = ixc[p];
        scol[p] = (w[ii] / dcol[ii]) / t;
    }
}

} // namespace

namespace ase {
namespace kernels {

// =============================================================================
// Public entry: secular equation solve for one merge.
// =============================================================================
void dc_secular_solve(AseHandle *ws, int k, double rho, double *S) {
    cudaStream_t st = ws->stream;
    CUDA_CHECK(cudaMemsetAsync(ws->dc_info, 0, sizeof(int), st));
    dc_laed4_kernel<<<div_up(k, 128), 128, 0, st>>>(
        k, ws->dc_dlamda, ws->dc_w, rho, ws->Sdc, ws->dc_tau, ws->dc_org, ws->dc_lam, ws->dc_info);
    if (k > 2) {
        const dim3 g(div_up(k, 256), std::min(k, 1024));
        dc_fill_delta_kernel<<<g, 256, 0, st>>>(k, ws->dc_dlamda, ws->dc_tau, ws->dc_org, ws->Sdc);
        dc_weights_kernel<<<div_up(k, 256), 256, 0, st>>>(k, ws->dc_dlamda, ws->Sdc, ws->dc_w,
                                                          ws->dc_wt);
    }
    dc_form_s_kernel<256><<<k, 256, 0, st>>>(k, ws->Sdc, ws->dc_wt, ws->dc_ixc, S, k > 2 ? 1 : 0);
}

} // namespace kernels
} // namespace ase
