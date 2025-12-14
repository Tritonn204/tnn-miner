#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>
#include <atomic>

constexpr size_t XELIS_TEMPLATE_SIZE = 112;
constexpr size_t XELIS_HASH_SIZE = 32;
constexpr size_t XELIS_MEMORY_SIZE_V3 = 531 * 128;

namespace Xelis_HIP {
    constexpr int MAX_NONCES = 16;

    void xelisHash_wrapper(
        int blocks,
        uint8_t* d_input,
        uint8_t* d_outputs,
        uint64_t* d_scratch,
        uint64_t* d_nonceBuffer,
        int* d_nonceCount,
        uint64_t nonceBase,
        uint64_t kernelIndex,
        uint32_t batchSize,
        size_t sharedMem,
        int deviceId,
        bool devMine
    );

    template<bool devMine>
    void copyWork(uint8_t* d_input, const uint8_t* work, int deviceId);

    void copyDiff(const uint8_t* diffBytes, bool devMine, int deviceId);

    void nonceCounter(int* d_nonceCount, int* h_nonceCount, 
                      uint64_t* d_nonceBuffer, uint64_t* h_nonceBuffer);

    void getHashAtIndex(uint8_t* d_outputs, int index, uint8_t* h_hash);

    size_t calcSharedMem(int blockSize);
}

namespace Xelis_HIP_Worker {
    struct workerCtx {
        int GPUCount;
        std::vector<int> blocks;
        std::vector<uint32_t> batchSizes;
        std::vector<size_t> sharedMem;

        // Per-GPU device pointers
        std::vector<uint8_t*> d_input;
        std::vector<uint8_t*> d_input_dev;
        std::vector<uint8_t*> d_outputs;
        std::vector<uint64_t*> d_scratch;
        std::vector<uint64_t*> d_nonceBuffer;
        std::vector<int*> d_nonceCount;

        uint8_t* cmpDiff;
        uint8_t* cmpDiff_dev;
    };

    void newCtx(workerCtx& ctx);
    void ctxMalloc(workerCtx& ctx);
    void ctxFree(workerCtx& ctx);
    void setDevice(int d);
}

void mineXelis_hip(int tid);