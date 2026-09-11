#include <stdint.h>
extern "C" __global__ [aicore] void row_softmax(__gm__ float*, __gm__ float*);
extern "C" void launch(void* input, void* output, void* stream) {
    row_softmax<<<4, nullptr, stream>>>((float*)input, (float*)output);
}
