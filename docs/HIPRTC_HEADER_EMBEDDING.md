# HIPRTC Header Embedding System

## Overview

HIPRTC (HIP Runtime Compilation) compiles GPU kernels at runtime, but it doesn't have access to the filesystem like a normal compiler. When your kernel code includes headers like:

```cpp
#include "blake3-inline.hip.inc"
#include "aes-inline.hip.inc"
```

HIPRTC can't find those files. This system solves that problem by **embedding header sources at compile-time** and providing them to HIPRTC as strings.

## How It Works

### 1. Compile-Time Embedding (CMake)

At CMake configuration time, we read header files and embed their contents as C++ string literals in a generated header file.

**Input:** Source files on disk
```
src/tnn_hip/crypto/blake3-inline.hip.inc
src/tnn_hip/crypto/aes-inline.hip.inc
```

**Output:** Generated C++ header
```cpp
// build/generated/xelis_embedded_headers.hpp
namespace hip_embedded {
    constexpr std::string_view SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_SOURCE = R"EMBEDSRC(
    // ... entire blake3-inline.hip.inc contents ...
    )EMBEDSRC";

    constexpr std::string_view SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_PATH =
        "src/tnn_hip/crypto/blake3-inline.hip.inc";
}
```

### 2. Build-Time Usage (C++)

The generated header is included in `hip_algo_registry.hpp` and used to populate `AlgoConfig`:

```cpp
#include "xelis_embedded_headers.hpp"

inline AlgoConfig XELIS_V3_CONFIG = {
    .rtc_headers = {
        {hip_embedded::SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_PATH,
         hip_embedded::SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_SOURCE},
        // ... more headers ...
    },
    // ... other config ...
};
```

### 3. Runtime Usage (HIPRTC)

When compiling kernels at runtime, we pass the embedded sources to HIPRTC:

```cpp
// Prepare header arrays
std::vector<const char*> header_sources;
std::vector<const char*> header_names;

for (const auto& header : config.rtc_headers) {
    header_sources.push_back(header.source.data());
    header_names.push_back(header.name.data());
}

// Create RTC program with embedded headers
hiprtcCreateProgram(
    &prog,
    kernel_source.c_str(),
    "kernel.hip",
    header_sources.size(),
    header_sources.data(),
    header_names.data()
);
```

Now when the kernel does `#include "blake3-inline.hip.inc"`, HIPRTC finds it in the embedded headers!

## Architecture Components

### 1. CMake Function (`cmake/embed_hip_sources.cmake`)

```cmake
embed_hip_sources(
    OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/xelis_embedded_headers.hpp"
    SOURCES
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/aes-inline.hip.inc"
)
```

**What it does:**
- Reads each source file
- Creates a valid C++ identifier from the path
- Embeds content using C++11 raw string literals
- Generates both `_SOURCE` (content) and `_PATH` (filename) constants
- Creates a manifest array for runtime enumeration

**Naming Convention:**
```
Path: src/tnn_hip/crypto/blake3-inline.hip.inc
→ Variable: SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC
```

### 2. AlgoConfig Structure (`src/tnn_hip/common/gpu_algo.hpp`)

```cpp
struct RTCHeader {
    std::string_view name;      // Include path as it appears in #include
    std::string_view source;    // Actual source code (embedded)
};

struct AlgoConfig {
    // ... other fields ...
    std::vector<RTCHeader> rtc_headers;   // Embedded headers for HIPRTC
};
```

### 3. Algorithm Registration (`src/tnn_hip/common/hip_algo_registry.hpp`)

Algorithms declare their RTC dependencies:

```cpp
inline AlgoConfig XELIS_V3_CONFIG = {
    .rtc_headers = {
        {hip_embedded::BLAKE3_PATH, hip_embedded::BLAKE3_SOURCE},
        {hip_embedded::AES_PATH, hip_embedded::AES_SOURCE},
    },
};
```

### 4. Runtime Compilation (Implementation in algorithm classes)

The `GPUAlgorithm` class uses the embedded headers when compiling:

```cpp
bool GPUAlgorithm::compile_kernel() {
    auto& config = get_config();

    // Read main kernel source from disk
    std::string kernel_source = read_file(config.source_path);

    // Extract embedded headers
    std::vector<const char*> header_sources;
    std::vector<const char*> header_names;

    for (const auto& header : config.rtc_headers) {
        header_sources.push_back(header.source.data());
        header_names.push_back(header.name.data());
    }

    // Create and compile RTC program
    hiprtcProgram prog;
    hiprtcCreateProgram(&prog, kernel_source.c_str(), "kernel.hip",
                       header_sources.size(),
                       header_sources.data(),
                       header_names.data());

    hiprtcCompileProgram(prog, num_opts, opts);
    // ... rest of compilation ...
}
```

## Adding New Headers

### Step 1: Update CMake

Edit `cmake/hip-crypto/<algo>/<algo>-hip.cmake`:

```cmake
embed_hip_sources(
    OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/<algo>_embedded_headers.hpp"
    SOURCES
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/new-header.hip.inc"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/another-header.hip"
)
```

### Step 2: Include Generated Header

In `hip_algo_registry.hpp`:

```cpp
#ifdef TNN_<ALGO>
#include "<algo>_embedded_headers.hpp"
#endif
```

### Step 3: Add to AlgoConfig

```cpp
inline AlgoConfig MY_ALGO_CONFIG = {
    .rtc_headers = {
        {hip_embedded::NEW_HEADER_PATH, hip_embedded::NEW_HEADER_SOURCE},
    },
};
```

### Step 4: Rebuild

```bash
cmake --build build
```

CMake will regenerate the embedded headers automatically!

## Example: Xelis V3

### Files Involved

**CMake Configuration:**
```cmake
# cmake/hip-crypto/xelis-hash/xelis-hash-hip.cmake
embed_hip_sources(
    OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/xelis_embedded_headers.hpp"
    SOURCES
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/blake3-inline.hip.inc"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/aes-inline.hip.inc"
        "${PROJECT_SOURCE_DIR}/src/tnn_hip/crypto/xelis-hash/xelis-hash-v3.hip"
)
```

**Generated Output:**
```cpp
// build/generated/xelis_embedded_headers.hpp
namespace hip_embedded {
    constexpr std::string_view SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_SOURCE = R"EMBEDSRC(
    /* ... blake3 implementation ... */
    )EMBEDSRC";

    constexpr std::string_view SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_PATH =
        "src/tnn_hip/crypto/blake3-inline.hip.inc";

    // ... AES and xelis-hash-v3 similarly ...
}
```

**Registry Configuration:**
```cpp
// src/tnn_hip/common/hip_algo_registry.hpp
#include "xelis_embedded_headers.hpp"

inline AlgoConfig XELIS_V3_CONFIG = {
    .rtc_headers = {
        {hip_embedded::SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_PATH,
         hip_embedded::SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_SOURCE},
        {hip_embedded::SRC_TNN_HIP_CRYPTO_AES_INLINE_HIP_INC_PATH,
         hip_embedded::SRC_TNN_HIP_CRYPTO_AES_INLINE_HIP_INC_SOURCE},
        {hip_embedded::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_PATH,
         hip_embedded::SRC_TNN_HIP_CRYPTO_XELIS_HASH_XELIS_HASH_V3_HIP_SOURCE}
    },
};
```

## Benefits

✅ **No Manual Copy-Paste** - Headers are automatically embedded
✅ **Always In Sync** - Regenerated on every build
✅ **Clean Separation** - Source stays in version control, generated code in build dir
✅ **Scalable** - Easy to add more headers
✅ **Type-Safe** - Using `std::string_view` ensures no copies
✅ **Compile-Time** - Zero runtime overhead

## Debugging

### Check Generated File

```bash
cat build/generated/xelis_embedded_headers.hpp
```

Look for:
- Correct namespace (`hip_embedded`)
- Variable naming matches expectations
- Content is properly embedded in raw string literals

### Verify CMake Ran

Look for CMake output:
```
-- Embedding HIP source: src/tnn_hip/crypto/blake3-inline.hip.inc -> SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC
-- Generated embedded headers file: build/generated/xelis_embedded_headers.hpp
```

### Common Issues

**Issue:** Variable name not found
```cpp
error: 'SRC_TNN_HIP_CRYPTO_BLAKE3_INLINE_HIP_INC_SOURCE' was not declared
```
**Solution:** Check the generated file - variable name might be different due to path transformation

**Issue:** Headers not regenerated after source change
**Solution:** Run `cmake --build build` - it regenerates on every build

**Issue:** HIPRTC compilation fails with "file not found"
**Solution:** Check that header name in `rtc_headers` matches the include path in the kernel exactly

## Advanced: Custom Include Paths

If your kernel uses a different include syntax:

**Kernel code:**
```cpp
#include <crypto/blake3.h>
```

**Set the path accordingly:**
```cpp
{{"crypto/blake3.h", hip_embedded::BLAKE3_SOURCE}
```

The path string must match what appears in the `#include` directive!

## Future Enhancements

Potential improvements:
- Recursive header dependency tracking
- Automatic dependency detection from source parsing
- Support for system headers (like `<cstdint>`)
- Compression for very large headers
- Hash-based change detection to avoid unnecessary rebuilds

## Related Documentation

- [CPU Mining Architecture](CPU_MINING_ARCHITECTURE.md) - Parallel CPU system design
- [GPU Mining Architecture](GPU_MINING_ARCHITECTURE.md) - GPU mining with HIPRTC (TODO)
- CMake Documentation: [file(READ)](https://cmake.org/cmake/help/latest/command/file.html#read)
- HIPRTC Documentation: [hiprtcCreateProgram](https://rocm.docs.amd.com/projects/HIPRTC/en/latest/index.html)
