#pragma once
// SEH wrappers for Orochi/HIP API calls on Windows.
//
// The AMD HIP runtime (amdhip64_7.dll) raises benign SEH exceptions during
// kernel launches and synchronisation calls. With /EHsc (synchronous C++
// exceptions only) these are fatal. Wrapping individual calls in
// __try/__except lets them propagate harmlessly.
//
// These are kept as small, standalone functions so that ROCm 6.4's clang-cl
// (which ICEs on /EHa or on large functions with inline __try blocks) can
// compile them without issues.

#include <Orochi/Orochi.h>

#if defined(_MSC_VER)
#include <windows.h>
#endif

// ---------------------------------------------------------------------------
// Device context / allocation
// ---------------------------------------------------------------------------
static inline oroError_t oro_safe_set_device(int device_id) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroSetDevice(device_id); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroSetDevice(device_id);
#endif
}

static inline oroError_t oro_safe_malloc(oroDeviceptr* ptr, size_t size) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroMalloc(ptr, size); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroMalloc(ptr, size);
#endif
}

static inline oroError_t oro_safe_free(oroDeviceptr ptr) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroFree(ptr); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroFree(ptr);
#endif
}

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------
static inline oroError_t oro_safe_launch(
    oroFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, oroStream_t stream,
    void** kernelParams, void** extra)
{
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try {
        err = oroModuleLaunchKernel(
            f, gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, stream, kernelParams, extra);
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        // Benign SEH from HIP runtime — swallow it; err keeps the real status.
    }
    return err;
#else
    return oroModuleLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, stream, kernelParams, extra);
#endif
}

// ---------------------------------------------------------------------------
// Event / sync wrappers
// ---------------------------------------------------------------------------
static inline oroError_t oro_safe_event_sync(oroEvent_t event) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroEventSynchronize(event); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroEventSynchronize(event);
#endif
}

static inline oroError_t oro_safe_event_query(oroEvent_t event) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroEventQuery(event); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroEventQuery(event);
#endif
}

static inline oroError_t oro_safe_event_record(oroEvent_t event, oroStream_t stream) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroEventRecord(event, stream); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroEventRecord(event, stream);
#endif
}

static inline oroError_t oro_safe_stream_sync(oroStream_t stream) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroStreamSynchronize(stream); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroStreamSynchronize(stream);
#endif
}

static inline oroError_t oro_safe_memcpy(
    void* dst, const void* src, size_t sizeBytes, oroMemcpyKind kind)
{
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroMemcpy(dst, src, sizeBytes, kind); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroMemcpy(dst, src, sizeBytes, kind);
#endif
}

static inline oroError_t oro_safe_memset(void* dst, int value, size_t sizeBytes) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroMemset(dst, value, sizeBytes); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroMemset(dst, value, sizeBytes);
#endif
}

static inline oroError_t oro_safe_event_elapsed(float* ms, oroEvent_t start, oroEvent_t stop) {
#if defined(_MSC_VER)
    oroError_t err = oroErrorUnknown;
    __try { err = oroEventElapsedTime(ms, start, stop); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroEventElapsedTime(ms, start, stop);
#endif
}

static inline oroError_t oro_safe_get_last_error() {
#if defined(_MSC_VER)
    oroError_t err = oroSuccess;
    __try { err = oroGetLastError(); }
    __except(EXCEPTION_EXECUTE_HANDLER) {}
    return err;
#else
    return oroGetLastError();
#endif
}
