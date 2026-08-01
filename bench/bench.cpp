/**
 * @file   bench.cpp
 * @brief  Benchmark ase::solve_ev over matrix sizes, reporting best-of-N time and GFLOPS.
 *
 * Defaults to the device path (solve_ev_d, matrix resident on the GPU); --host times the
 * host-pointer wrapper, which includes a full host↔device round trip. GFLOPS uses the
 * standard full symmetric eigendecomposition count (22/3)·n³ (JobZ='V', as in LAPACK).
 *
 * Usage: ase_bench [size...] [--iters N] [--host]
 *   default sizes: 512 1024 2048 4096 8192 16384 32768
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#include <ase/ase.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

namespace
{

// Deterministic symmetric matrix with entries in (-1, 1) — lower triangle mirrored to
// the upper so A is exactly symmetric.
void make_symmetric(std::vector<double>& A, int n)
{
    unsigned s = 1;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return ((double)(s >> 8) / 16777216.0) * 2.0 - 1.0;
    };
    for (int c = 0; c < n; ++c)
        for (int r = c; r < n; ++r) {
            const double v = rnd();
            A[(size_t)c * n + r] = v;
            A[(size_t)r * n + c] = v;
        }
}

void usage(const char* argv0)
{
    fprintf(stderr,
            "usage: %s [size...] [--iters N] [--host]\n"
            "  --host   time the host-pointer solution (includes host↔device copies)\n"
            "  default sizes: 512 1024 2048 4096 8192 16384 32768\n",
            argv0);
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<int> sizes;
    int iters = 3;
    bool host = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--host") == 0)
            host = true;
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else
            sizes.push_back(std::atoi(argv[i]));
    }
    if (sizes.empty()) sizes = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    if (iters < 1) iters = 1;

    ase::AseHandle* ws = ase::handle_alloc(0);
    printf("%-8s %10s %10s %12s\n", "n", host ? "h_time_ms" : "d_time_ms", "GFLOPS", "mem_MB");
    for (int n : sizes) {
        const size_t bytes = (size_t)n * n * sizeof(double);
        std::vector<double> A((size_t)n * n), eval(n), evec((size_t)n * n);
        make_symmetric(A, n);

        // device-resident input for the solve_ev_d path
        double* dA = nullptr;
        double* deval = nullptr;
        double* devec = nullptr;
        if (!host) {
            if (cudaMalloc(&dA, bytes) != cudaSuccess) { // e.g. this size spills the GPU
                printf("%-8d %10s\n", n, "skip (OOM)");
                continue;
            }
            cudaMalloc(&deval, (size_t)n * sizeof(double));
            cudaMalloc(&devec, bytes);
            cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
        }

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            cudaEventRecord(t0);
            if (host)
                ase::solve_ev(ws, A.data(), n, eval.data(), evec.data());
            else
                ase::solve_ev_d(ws, dA, n, deval, devec);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, t0, t1);
            best = std::min(best, (double)ms);
        }
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        if (!host) {
            cudaFree(dA);
            cudaFree(deval);
            cudaFree(devec);
        }

        const double flops = (22.0 / 3.0) * (double)n * n * n;
        printf("%-8d %10.3f %10.0f %12.1f\n", n, best, flops / (best * 1e-3) / 1e9,
               bytes / (1024.0 * 1024.0));
    }
    ase::handle_free(ws);
    return 0;
}
