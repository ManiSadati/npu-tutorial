#include <stdint.h>
// Raw A5 CCE intrinsics. Compile this .cpp with ccec -x cce.
// Fixed geometry: one block, 256 FP32 elements, four 64-lane vectors.
extern "C" __global__ [aicore] void vector_add(__gm__ float* a, __gm__ float* b, __gm__ float* c) {
    auto ua = (__ubuf__ float*)get_imm(0);
    auto ub = (__ubuf__ float*)get_imm(1024);
    auto uc = (__ubuf__ float*)get_imm(2048);
    // DMA lengths are bytes; one contiguous 1024-byte burst.
    copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)ua, (__gm__ uint16_t*)a,
                           0, 1, 1024, 0, 0, false, 0, 0, 0);
    copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)ub, (__gm__ uint16_t*)b,
                           0, 1, 1024, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    __VEC_SCOPE__ {
        vector_bool mask = pset_b32(PAT_ALL);
        for (uint16_t i = 0; i < 4; ++i) {
            vector_f32 va, vb, vc;
            vlds(va, ua, i * 64, NORM);
            vlds(vb, ub, i * 64, NORM);
            vadd(vc, va, vb, mask, MODE_ZEROING);
            vsts(vc, uc, i * 64, NORM_B32, mask);
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_v2((__gm__ uint16_t*)c, (__ubuf__ uint16_t*)uc,
                           0, 1, 1024, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
}
