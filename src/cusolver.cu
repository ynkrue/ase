/**
 * @file   cusolver.cu
 * @brief  cuSOLVER wrappers — ase::cusolver namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "handle.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>

namespace ase
{
namespace cusolver
{

void geqrf(AseHandle* ws, int m, int n, double* A, int lda, double* tau)
{
    const int slot = ws->info_used < ws->info_cap ? ws->info_used++ : ws->info_cap - 1;
    int*      info = ws->d_info + slot;

    CUSOLVER_CHECK(cusolverDnDgeqrf(ws->cusolver, m, n, A, lda, tau, ws->geqrf_buf, ws->geqrf_lwork, info));
}

void geqrf_check(AseHandle* ws)
{
    if (ws->info_used == 0) return;

    CUDA_CHECK(cudaMemcpyAsync(ws->h_info, ws->d_info, (size_t)ws->info_used * sizeof(int), cudaMemcpyDeviceToHost,
                               ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    for (int i = 0; i < ws->info_used; ++i)
    {
        if (ws->h_info[i] != 0)
        {
            fprintf(stderr, "ase: geqrf failed on panel %d, info = %d\n", i, ws->h_info[i]);
            exit(EXIT_FAILURE);
        }
    }
}

} // namespace cusolver
} // namespace ase
