#!/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: $0 <TNN_VERSION> [amd|nvidia|cpu|all]"
    exit 1
fi

TNN_VERSION=$1
TARGET="${2:-all}"   # default to "all" if not provided

# Function to run cmake and build only if cmake succeeds
build_target() {
    local target_dir=$1
    local hip_flag=$2
    local hip_platform=$3

    local build_dir="./hip-build/linux/${target_dir}"

    # Ensure build dir exists
    mkdir -p "$build_dir"

    # Remove CMakeCache.txt to refresh cache for each target
    rm -f "${build_dir}/CMakeCache.txt"

    # Run cmake command
    HIP_PLATFORM=$hip_platform cmake -S . -B "$build_dir" \
        -DCMAKE_PREFIX_PATH="$HIP_PATH" \
        -DWITH_HIP=$hip_flag \
        -DHIP_PLATFORM=$hip_platform \
        -DTNN_VERSION=$TNN_VERSION

    if [ $? -ne 0 ]; then
        echo "CMake failed for target '${target_dir}', skipping build."
        return 1
    fi

    # If cmake is successful, run the build command
    cmake --build "$build_dir" --target all -- -j"$(nproc)"
    if [ $? -ne 0 ]; then
        echo "Build failed for target '${target_dir}'."
        return 1
    fi

    return 0
}

# Only need HIP vars when doing HIP builds; still fine to export always
export HIP_PATH="$(hipconfig --path 2>/dev/null)"
export ROCM_PATH="$(hipconfig --rocmpath 2>/dev/null)"

# Dispatch based on TARGET
case "$TARGET" in
    (amd)
        echo "Building for AMD (ROCm)..."
        build_target "amd" ON amd || echo "Failed to build for AMD."
        ;;
    (nvidia)
        echo "Building for NVIDIA (HIP on CUDA)..."
        build_target "nvidia" ON nvidia || echo "Failed to build for NVIDIA."
        ;;
    (cpu)
        echo "Building CPU-only..."
        build_target "cpu" OFF "" || echo "Failed to build for CPU-only."
        ;;
    (all)
        echo "Building for AMD (ROCm)..."
        build_target "amd" ON amd || echo "Failed to build for AMD."

        echo "Building for NVIDIA (HIP on CUDA)..."
        build_target "nvidia" ON nvidia || echo "Failed to build for NVIDIA."

        echo "Building CPU-only..."
        build_target "cpu" OFF "" || echo "Failed to build for CPU-only."
        ;;
    (*)
        echo "Invalid target: '$TARGET'"
        echo "Usage: $0 <TNN_VERSION> [amd|nvidia|cpu|all]"
        exit 1
        ;;
esac
