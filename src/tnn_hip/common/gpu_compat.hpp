// gpu_compat.hpp — Unified GPU API header
//
// WITH_OROCHI: includes Orochi (runtime HIP/CUDA dispatch via dlopen)
// Otherwise:   includes HIP headers and aliases hip* -> oro* names
#pragma once

#ifdef WITH_OROCHI

#include <Orochi/Orochi.h>

#else // !WITH_OROCHI — standard HIP, alias hip* to oro* so code uses oro* everywhere

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

// ---- Types ----
using oroError_t     = hipError_t;
using oroModule_t    = hipModule_t;
using oroFunction_t  = hipFunction_t;
using oroEvent_t     = hipEvent_t;
using oroStream_t    = hipStream_t;
using oroDeviceptr_t = hipDeviceptr_t;
using oroDeviceptr   = oroDeviceptr_t;
using oroDevice_t    = hipDevice_t;
using oroCtx_t       = hipCtx_t;
using oroDeviceProp_t = hipDeviceProp_t;
using orortcProgram  = hiprtcProgram;
using orortcResult   = hiprtcResult;

// ---- Constants ----
constexpr auto oroSuccess           = hipSuccess;
constexpr auto oroErrorNotReady     = hipErrorNotReady;
constexpr auto oroErrorNoDevice     = hipErrorNoDevice;
constexpr auto oroMemcpyHostToDevice  = hipMemcpyHostToDevice;
constexpr auto oroMemcpyDeviceToHost  = hipMemcpyDeviceToHost;
constexpr auto oroEventDefault      = hipEventDefault;
constexpr auto ORORTC_SUCCESS       = HIPRTC_SUCCESS;

// ---- Runtime API ----
#define oroMalloc               hipMalloc
#define oroFree                 hipFree
#define oroMemcpy               hipMemcpy
#define oroMemcpyToSymbol       hipMemcpyToSymbol
#define oroMemcpyFromSymbol     hipMemcpyFromSymbol
#define oroMemset               hipMemset
#define oroMemsetAsync          hipMemsetAsync
#define oroMemGetInfo           hipMemGetInfo

#define oroGetDeviceCount       hipGetDeviceCount
#define oroGetDevice            hipGetDevice
#define oroSetDevice            hipSetDevice
#define oroGetDeviceProperties  hipGetDeviceProperties
#define oroDeviceGetPCIBusId    hipDeviceGetPCIBusId
#define oroDeviceReset          hipDeviceReset
#define oroDeviceSynchronize    hipDeviceSynchronize

#define oroModuleLoadData       hipModuleLoadData
#define oroModuleGetFunction    hipModuleGetFunction
#define oroModuleLaunchKernel   hipModuleLaunchKernel
#define oroModuleUnload         hipModuleUnload

#define oroEventCreate          hipEventCreate
#define oroEventCreateWithFlags hipEventCreateWithFlags
#define oroEventDestroy         hipEventDestroy
#define oroEventRecord          hipEventRecord
#define oroEventSynchronize     hipEventSynchronize
#define oroEventQuery           hipEventQuery
#define oroEventElapsedTime     hipEventElapsedTime

#define oroStreamCreate         hipStreamCreate
#define oroStreamDestroy        hipStreamDestroy
#define oroStreamSynchronize    hipStreamSynchronize

// oroGetErrorString: Orochi uses 2-arg output param; wrap HIP's 1-arg return style
inline oroError_t oroGetErrorString(oroError_t err, const char** pStr) {
    *pStr = hipGetErrorString(err);
    return oroSuccess;
}
#define oroGetLastError         hipGetLastError

#define oroOccupancyMaxActiveBlocksPerMultiprocessor \
        hipOccupancyMaxActiveBlocksPerMultiprocessor

#define oroLaunchKernel         hipLaunchKernel

// ---- HIPRTC ----
#define orortcCreateProgram     hiprtcCreateProgram
#define orortcCompileProgram    hiprtcCompileProgram
#define orortcGetCodeSize       hiprtcGetCodeSize
#define orortcGetCode           hiprtcGetCode
#define orortcGetProgramLogSize hiprtcGetProgramLogSize
#define orortcGetProgramLog     hiprtcGetProgramLog
#define orortcDestroyProgram    hiprtcDestroyProgram

#endif // WITH_OROCHI

// ---- Device handle helper ----
// Orochi wraps device ordinals in oroDevice (encodes backend API).
// Standard HIP uses raw int. This helper lets all code use ordinals uniformly.
#ifdef WITH_OROCHI
inline oroDevice tnn_get_device(int ordinal) {
    oroDevice dev;
    oroDeviceGet(&dev, ordinal);
    return dev;
}
#else
inline oroDevice_t tnn_get_device(int ordinal) { return ordinal; }
#endif

// ---- Convenience: returns error string directly ----
inline const char* tnn_error_string(oroError_t err) {
    const char* str = nullptr;
    oroGetErrorString(err, &str);
    return str ? str : "unknown error";
}

// ---- Platform query ----
// Under Orochi, s_api is thread_local and set by the active context,
// so this returns the vendor for whichever device context the calling thread holds.
// Under direct HIP, the platform is fixed at compile time.
#ifdef WITH_OROCHI
inline bool tnn_is_amd_device()    { return oroGetCurAPI(0) == ORO_API_HIP; }
inline bool tnn_is_nvidia_device() { return oroGetCurAPI(0) == ORO_API_CUDA; }
#else
inline bool tnn_is_amd_device() {
#ifdef __HIP_PLATFORM_AMD__
    return true;
#else
    return false;
#endif
}
inline bool tnn_is_nvidia_device() {
#ifdef __HIP_PLATFORM_NVIDIA__
    return true;
#else
    return false;
#endif
}
#endif
