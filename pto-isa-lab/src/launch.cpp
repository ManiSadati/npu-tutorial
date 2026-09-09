#include <stdint.h>
extern "C" __global__ [aicore] void vector_add(__gm__ float*, __gm__ float*, __gm__ float*);
extern "C" void launch(void* a, void* b, void* c, void* stream) {
    vector_add<<<1, nullptr, stream>>>((float*)a, (float*)b, (float*)c);
}
