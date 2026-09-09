#include <pto/pto-inst.hpp>
using namespace pto;
// One vector block, 256 FP32 elements; each UB allocation occupies 1024 bytes.
extern "C" __global__ AICORE void vector_add(__gm__ float* a, __gm__ float* b, __gm__ float* c) {
    using G = GlobalTensor<float, Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>>;
    G ga(a), gb(b), gc(c);
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256> ta, tb, tc;
    TASSIGN(ta, 0); TASSIGN(tb, 1024); TASSIGN(tc, 2048);
    TLOAD(ta, ga); TLOAD(tb, gb);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(tc, ta, tb);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gc, tc);
    pipe_barrier(PIPE_ALL);
}
