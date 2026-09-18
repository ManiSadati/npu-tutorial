#include <stdint.h>
// 20x1024 FP32, one vector block, 1 batch(es) of 20 rows.
extern "C" __global__ [aicore] void row_softmax(__gm__ float* input, __gm__ float* output) {
    auto x = (__ubuf__ float*)get_imm(0);
    auto ex = (__ubuf__ float*)get_imm(81920);
    auto y = (__ubuf__ float*)get_imm(163840);
    // Rows 0..19; reuse the same UB allocation.
    copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)x, (__gm__ uint16_t*)(input + 0),
                           0, 1, 81920, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    __VEC_SCOPE__ {
        vector_bool mask = pset_b32(PAT_ALL);
        for (uint16_t row = 0; row < 20; ++row) {
            vector_f32 maxv, value, reduced, rowmax, sums, expv, rowsum, result;
            vdup(maxv, -3.402823466e38f, mask, MODE_ZEROING);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(value, x, row * 1024 + i * 64, NORM);
                vmax(maxv, maxv, value, mask, MODE_ZEROING);
            }
            vcmax(reduced, maxv, mask, MODE_ZEROING);
            vdup(rowmax, reduced, mask, POS_LOWEST, MODE_ZEROING);
            vdup(sums, 0.0f, mask, MODE_ZEROING);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(value, x, row * 1024 + i * 64, NORM);
                vsub(value, value, rowmax, mask, MODE_ZEROING);
                vexp(expv, value, mask, MODE_ZEROING);
                vsts(expv, ex, row * 1024 + i * 64, NORM_B32, mask);
                vadd(sums, sums, expv, mask, MODE_ZEROING);
            }
            vcadd(reduced, sums, mask, MODE_ZEROING);
            vdup(rowsum, reduced, mask, POS_LOWEST, MODE_ZEROING);
            mem_bar(VST_VLD);
            for (uint16_t i = 0; i < 16; ++i) {
                vlds(expv, ex, row * 1024 + i * 64, NORM);
                vdiv(result, expv, rowsum, mask, MODE_ZEROING);
                vsts(result, y, row * 1024 + i * 64, NORM_B32, mask);
            }
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_v2((__gm__ uint16_t*)(output + 0), (__ubuf__ uint16_t*)y,
                           0, 1, 81920, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
}
