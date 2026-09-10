#include <stdint.h>
extern "C" __global__ [aicore] void matrix_add(__gm__ float*, __gm__ float*, __gm__ float*);
extern "C" void launch(void* a, void* b, void* c, void* stream) {
    // Exactly one AIV block: all 100 rows execute on the same vector core.
    matrix_add<<<1, nullptr, stream>>>((float*)a, (float*)b, (float*)c);
}
