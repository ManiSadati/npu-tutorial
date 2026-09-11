#include <stdint.h>
// Raw CCE: stable FP32 row softmax. One row uses sixteen 64-lane vectors.
extern "C" __global__ [aicore] void row_softmax(__gm__ float* input, __gm__ float* output) {
    auto x = (__ubuf__ float*)get_imm(0);
    auto ex = (__ubuf__ float*)get_imm(4096);
    auto y = (__ubuf__ float*)get_imm(8192);
    auto scalar = (__ubuf__ float*)get_imm(12288);
    for (unsigned row = get_block_idx(); row < 100; row += 4) {
        copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)x, (__gm__ uint16_t*)(input + row * 1024),
                               0, 1, 4096, 0, 0, false, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        __VEC_SCOPE__ {
            vector_bool mask = pset_b32(PAT_ALL);
            vector_f32 maxv, value, reduced, rowmax, sums, expv, rowsum, result;
            vdup(maxv, -3.402823466e38f, mask, MODE_ZEROING);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(value, x, i * 64, NORM);
                vmax(maxv, maxv, value, mask, MODE_ZEROING);
            }
            vcmax(reduced, maxv, mask, MODE_ZEROING);
            vsts(reduced, scalar, 0, ONEPT_B32, mask);
            mem_bar(VST_VLD);
            vlds(rowmax, scalar, 0, BRC_B32);
            vdup(sums, 0.0f, mask, MODE_ZEROING);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(value, x, i * 64, NORM);
                vsub(value, value, rowmax, mask, MODE_ZEROING);
                vexp(expv, value, mask, MODE_ZEROING);
                vsts(expv, ex, i * 64, NORM_B32, mask);
                vadd(sums, sums, expv, mask, MODE_ZEROING);
            }
            vcadd(reduced, sums, mask, MODE_ZEROING);
            // The sum occupies a different 32-byte slot from the maximum.
            vsts(reduced, scalar, 8, ONEPT_B32, mask);
            mem_bar(VST_VLD);
            vlds(rowsum, scalar, 8, BRC_B32);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(expv, ex, i * 64, NORM);
                vdiv(result, expv, rowsum, mask, MODE_ZEROING);
                vsts(result, y, i * 64, NORM_B32, mask);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_v2((__gm__ uint16_t*)(output + row * 1024), (__ubuf__ uint16_t*)y,
                               0, 1, 4096, 0, 0, 0);
        pipe_barrier(PIPE_ALL);
    }
}
