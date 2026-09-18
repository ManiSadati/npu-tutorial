#include <pto/pto-inst.hpp>
using namespace pto;
// Four vector blocks, owning 3, 3, 2 and 2 rows. All addresses are UB byte offsets.
extern "C" __global__ AICORE void row_softmax(__gm__ float* input, __gm__ float* output) {
    using G = GlobalTensor<float, Shape<1,1,1,1,1024>, pto::Stride<1024,1024,1024,1024,1>>;
    using Row = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024>;
    using Scalar = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1>;
    Row x, shifted, expx, y, scratch;
    Scalar maximum, sum;
    TASSIGN(x, 0); TASSIGN(shifted, 4096); TASSIGN(expx, 8192);
    TASSIGN(y, 12288); TASSIGN(scratch, 16384);
    TASSIGN(maximum, 20480); TASSIGN(sum, 20512);
    for (unsigned row = get_block_idx(); row < 10; row += 4) {
        G gx(input + row * 1024), gy(output + row * 1024);
        TLOAD(x, gx);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TROWMAX(maximum, x, scratch); pipe_barrier(PIPE_ALL);
        TROWEXPANDSUB(shifted, x, maximum); pipe_barrier(PIPE_ALL);
        TEXP(expx, shifted); pipe_barrier(PIPE_ALL);
        TROWSUM(sum, expx, scratch); pipe_barrier(PIPE_ALL);
        TROWEXPANDDIV(y, expx, sum);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(gy, y);
        // Finish the store before reusing this row's UB buffers.
        pipe_barrier(PIPE_ALL);
    }
}
