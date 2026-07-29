/**
 * @file   cusolver.cu
 * @brief  Type-dispatching cuSOLVER wrappers — ase::cusolver namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "handle.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace ase {
namespace cusolver {

namespace {
inline void check_info([[maybe_unused]] int *d_info, [[maybe_unused]] const char *name,
                       [[maybe_unused]] cudaStream_t stream) {
#ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "cuSOLVER %s failed: info = %d\n", name, h_info);
        exit(1);
    }
#endif
}
} // namespace

template <typename T>
void geqrf(SolverHandle<T> *ws, int m, int n, T *A, int lda, T *tau, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSgeqrf(ws->cusolver, m, n, A, lda, tau, ws->geqrf_buf,
                                        ws->geqrf_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(cusolverDnDgeqrf(ws->cusolver, m, n, A, lda, tau, ws->geqrf_buf,
                                        ws->geqrf_lwork, ws->d_info));
    check_info(ws->d_info, "geqrf", stream);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void geqrf<T>(SolverHandle<T> *, int, int, T *, int, T *, cudaStream_t);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cusolver
} // namespace ase
