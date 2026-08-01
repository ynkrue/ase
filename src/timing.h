/**
 * @file   timing.h
 * @brief  Optional per-stage CUDA-event timing (internal).
 *
 * Active only while ws->timing is set (see ase_timing_enable). When off, the begin/end
 * calls are no-ops so the hot path carries no overhead.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#pragma once
#include "handle.h"
#include <cuda_runtime.h>

namespace ase {
namespace detail {

/// Begin timing the current stage on ws->stream. Returns a token event, or nullptr when
/// timing is off.
cudaEvent_t stage_begin(AseHandle* ws);

/// Stop @p ev0 and accumulate the stage's elapsed CUDA time into ws->timing_ms[idx].
/// No-op when @p ev0 is null.
void stage_end(AseHandle* ws, int idx, cudaEvent_t ev0);

} // namespace detail
} // namespace ase
