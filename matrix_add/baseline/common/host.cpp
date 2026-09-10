#include <acl/acl.h>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
extern "C" void launch(void*, void*, void*, void*);
static void check(aclError e, const char* op) {
    if (e != ACL_SUCCESS) { std::fprintf(stderr, "%s failed: %d\n", op, int(e)); throw std::runtime_error(op); }
}
#define ACL(op) check((op), #op)
int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "Usage: verify a.bin b.bin output.bin\n"); return 2; }
    constexpr size_t count = 10 * 512, bytes = count * sizeof(float);
    void *input = nullptr, *input_b = nullptr, *output = nullptr;
    aclrtStream stream = nullptr;
    bool initialized = false, device = false;
    int status = 1;
    try {
        std::vector<float> x(count), b(count), y(count, std::numeric_limits<float>::quiet_NaN());
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != std::streampos(bytes)) throw std::runtime_error("Input must be 10x512 FP32");
        file.seekg(0); file.read(reinterpret_cast<char*>(x.data()), bytes);
        if (!file) throw std::runtime_error("Input read failed");
        std::ifstream file_b(argv[2], std::ios::binary | std::ios::ate);
        if (!file_b || file_b.tellg() != std::streampos(bytes)) throw std::runtime_error("B must be 10x512 FP32");
        file_b.seekg(0); file_b.read(reinterpret_cast<char*>(b.data()), bytes);
        if (!file_b) throw std::runtime_error("B read failed");
        ACL(aclInit(nullptr)); initialized = true;
        ACL(aclrtSetDevice(0)); device = true;
        ACL(aclrtCreateStream(&stream));
        ACL(aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMalloc(&input_b, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMalloc(&output, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMemcpy(input, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL(aclrtMemcpy(output, bytes, y.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL(aclrtMemcpy(input_b, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        launch(input, input_b, output, stream);
        ACL(aclrtSynchronizeStream(stream));
        ACL(aclrtMemcpy(y.data(), bytes, output, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream result(argv[3], std::ios::binary);
        result.write(reinterpret_cast<const char*>(y.data()), bytes); result.close();
        if (!result) throw std::runtime_error("Output write failed");
        std::printf("Saved 10x512 FP32 output to %s\n", argv[3]);
        status = 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); }
    auto cleanup = [&](aclError e) { if (e != ACL_SUCCESS) { std::fprintf(stderr, "Cleanup failed: %d\n", int(e)); status = 1; } };
    if (output) cleanup(aclrtFree(output));
    if (input_b) cleanup(aclrtFree(input_b));
    if (input) cleanup(aclrtFree(input));
    if (stream) cleanup(aclrtDestroyStream(stream));
    if (device) cleanup(aclrtResetDevice(0));
    if (initialized) cleanup(aclFinalize());
    return status;
}
