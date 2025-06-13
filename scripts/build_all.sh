#!/bin/bash

# Check if TNN_VERSION is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <TNN_VERSION>"
    exit 1
fi

TNN_VERSION=$1

# Function to run cmake and build only if cmake succeeds
build_target() {
    local target_dir=$1
    local hip_flag=$2
    local hip_platform=$3

    # Remove CMakeCache.txt to refresh cache for each target
    rm -f "../bin/linux/$target_dir/CMakeCache.txt"

    # Run cmake command
    HIP_PLATFORM=$hip_platform cmake -S .. -B "../bin/linux/$target_dir" -DCMAKE_PREFIX_PATH="$HIP_PATH" -DWITH_HIP=$hip_flag -DHIP_PLATFORM=$hip_platform -DTNN_VERSION=$TNN_VERSION
    if [ $? -ne 0 ]; then
        echo "CMake failed, skipping build."
        return 1
    fi

    # If cmake is successful, run the build command
    cmake --build "../bin/linux/$target_dir" --target all -- -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Build failed"
        return 1
    fi

    return 0
}


export HIP_PATH="$(hipconfig --path)"
export ROCM_PATH="$(hipconfig --rocmpath)"

# Build for AMD using ROCm
build_target "" ON amd
if [ $? -ne 0 ]; then
    echo "Failed to build for AMD."
fi

# Build for NVIDIA using hipcc (with HIP_PLATFORM=nvidia)
build_target "" ON nvidia
if [ $? -ne 0 ]; then
    echo "Failed to build for NVIDIA."
fi

# Build without HIP
build_target "" OFF ""
if [ $? -ne 0 ]; then
    echo "Failed to build for CPU-only."
fi
