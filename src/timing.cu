/**
 * @file   timing.cu
 * @brief  Per-stage CUDA-event timers (see timing.h).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#include "timing.h"

namespace ase {
namespace detail {

cudaEvent_t stage_begin(AseHandle* ws)
{
    if (!ws->timing) return nullptr;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, ws->stream);
    return e;
}

void stage_end(AseHandle* ws, int idx, cudaEvent_t ev0)
{
    if (!ev0) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, ws->stream);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, e);
    ws->timing_ms[idx] += ms;
    cudaEventDestroy(ev0);
    cudaEventDestroy(e);
}

} // namespace detail
} // namespace ase
