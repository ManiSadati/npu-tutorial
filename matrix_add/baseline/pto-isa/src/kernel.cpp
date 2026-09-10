#include <pto/pto-inst.hpp>
using namespace pto;
// One vector block processes all 100 rows; three 2048-byte UB row buffers.
extern "C" __global__ AICORE void matrix_add(__gm__ float* a, __gm__ float* b, __gm__ float* c) {
    using G = GlobalTensor<float, Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>>;
    for (unsigned row = 0; row < 100; ++row) {
        G ga(a + row * 512), gb(b + row * 512), gc(c + row * 512);
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512> ta, tb, tc;
        TASSIGN(ta, 0); TASSIGN(tb, 2048); TASSIGN(tc, 4096);
        TLOAD(ta, ga); TLOAD(tb, gb);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(tc, ta, tb);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(gc, tc);
        pipe_barrier(PIPE_ALL);
    }
}
