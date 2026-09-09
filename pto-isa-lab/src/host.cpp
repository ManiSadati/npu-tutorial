#include <acl/acl.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
extern "C" void launch(void*, void*, void*, void*);
static void check(aclError e, const char* op) {
    if (e != ACL_SUCCESS) { std::fprintf(stderr, "%s failed: %d\n", op, int(e)); throw std::runtime_error(op); }
}
#define ACL(op) check((op), #op)
int main() {
    constexpr size_t n = 256, bytes = n * sizeof(float);
    void *a = nullptr, *b = nullptr, *c = nullptr;
    aclrtStream stream = nullptr;
    bool initialized = false, device = false;
    int status = 1;
    try {
        ACL(aclInit(nullptr)); initialized = true;
        ACL(aclrtSetDevice(0)); device = true;
        ACL(aclrtCreateStream(&stream));
        ACL(aclrtMalloc(&a, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMalloc(&b, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMalloc(&c, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        for (int trial = 0; trial < 3; ++trial) {
            std::array<float, n> x{}, y{}, out{};
            for (size_t i = 0; i < n; ++i) {
                x[i] = float(int(i) - 128) * 0.25f;
                y[i] = trial == 0 ? float(int(i % 17) - 8) * 0.5f :
                       trial == 1 ? -x[i] : 0.0f;
                out[i] = NAN;
            }
            ACL(aclrtMemcpy(a, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL(aclrtMemcpy(b, bytes, y.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL(aclrtMemcpy(c, bytes, out.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            launch(a, b, c, stream);
            ACL(aclrtSynchronizeStream(stream));
            ACL(aclrtMemcpy(out.data(), bytes, c, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
            size_t errors = 0;
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(out[i]) || out[i] != x[i] + y[i]) {
                    if (errors++ < 4) std::fprintf(stderr, "index=%zu got=%g expected=%g\n", i, out[i], x[i]+y[i]);
                }
            std::printf("Trial %d: %s (%zu/%zu correct)\n", trial, errors ? "FAIL" : "PASS", n-errors, n);
            if (errors) throw std::runtime_error("Numerical verification failed");
        }
        status = 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); }
    auto cleanup = [&](aclError e) { if (e != ACL_SUCCESS) { std::fprintf(stderr, "Cleanup failed: %d\n", int(e)); status = 1; } };
    if (c) cleanup(aclrtFree(c));
    if (b) cleanup(aclrtFree(b));
    if (a) cleanup(aclrtFree(a));
    if (stream) cleanup(aclrtDestroyStream(stream));
    if (device) cleanup(aclrtResetDevice(0));
    if (initialized) cleanup(aclFinalize());
    return status;
}
