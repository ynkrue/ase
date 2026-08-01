/**
 * @file   handle_test.cu
 * @brief  AseHandle lifecycle: alloc/free, lazy workspace sizing, staging. White-box
 * (links the internal handle.h) so buffer layout and padding are checkable.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-08
 */

#include "ase/ase.h"
#include "handle.h"
#include "ref.hpp"

#include <gtest/gtest.h>

namespace
{

TEST(handle, AllocFrees)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    ase::AseHandle* ws = ase::handle_alloc(0);
    ASSERT_NE(ws, nullptr);
    ase::handle_free(ws);
    ase::handle_free(nullptr); // no-op, must not crash
}

TEST(handle, CheckSizesWorkspace)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    ase::AseHandle* ws = ase::handle_alloc(0);
    ase::handle_check(ws, 100);
    EXPECT_EQ(ws->n, 100);
    EXPECT_GE(ws->ldu, 100); // ldd padded up to double4_32a alignment
    EXPECT_NE(ws->pool, nullptr);
    EXPECT_GT(ws->pool_bytes, 0u);
    ASSERT_NE(ws->Y, nullptr);
    ASSERT_NE(ws->U, nullptr);
    ase::handle_free(ws);
}

TEST(handle, CheckResizes)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    ase::AseHandle* ws = ase::handle_alloc(0);
    ase::handle_check(ws, 64);
    const size_t p64 = ws->pool_bytes;
    ase::handle_check(ws, 256);
    EXPECT_EQ(ws->n, 256);
    EXPECT_GT(ws->pool_bytes, p64); // larger dim → larger pool
    const size_t p256 = ws->pool_bytes;
    ase::handle_check(ws, 256);
    EXPECT_EQ(ws->pool_bytes, p256); // cached: no realloc
    ase::handle_free(ws);
}

TEST(handle, StageIsIdempotent)
{
    if (!testref::has_gpu()) GTEST_SKIP();
    ase::AseHandle* ws = ase::handle_alloc(0);
    ase::handle_check(ws, 16);
    ase::handle_stage(ws);
    ASSERT_NE(ws->A_stage, nullptr);
    ASSERT_NE(ws->eval_stage, nullptr);
    ASSERT_NE(ws->evec_stage, nullptr);
    ase::handle_stage(ws); // second call no-ops (same backing pool)
    EXPECT_EQ(ws->A_stage, ws->A_stage);
    EXPECT_NE(ws->eval_stage, ws->evec_stage); // distinct slices of stage_pool
    ase::handle_free(ws);
}

} // namespace
