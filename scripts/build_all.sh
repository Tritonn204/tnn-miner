#!/bin/bash

# Usage check
if [ -z "$1" ]; then
    echo "Usage: $0 <TNN_VERSION> [amd|nvidia|cpu|orochi|all]"
    exit 1
fi

TNN_VERSION=$1
TARGET="${2:-all}"   # default to "all" if not provided
DEBUG=0
if [ "$3" = "--debug" ] || [ "$3" = "-debug" ]; then
    DEBUG=1
fi

# Function to run cmake and build only if cmake succeeds
build_target() {
    local target_dir=$1
    shift
    # remaining args are cmake flags
    local cmake_flags=("$@")

    local build_dir="./hip-build/linux/${target_dir}"

    # Ensure build dir exists
    mkdir -p "$build_dir"

    # Remove CMakeCache.txt to refresh cache for each target
    rm -f "${build_dir}/CMakeCache.txt"

    # Add debug flags if requested
    if [ "$DEBUG" -eq 1 ]; then
        cmake_flags+=("-DCMAKE_BUILD_TYPE=Debug")
    fi

    # Run cmake command
    cmake -S . -B "$build_dir" \
        -DTNN_VERSION=$TNN_VERSION \
        "${cmake_flags[@]}"

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
        build_target "amd" \
            -DCMAKE_PREFIX_PATH="$HIP_PATH" \
            -DWITH_HIP=ON \
            -DHIP_PLATFORM=amd \
            || echo "Failed to build for AMD."
        ;;
    (nvidia)
        echo "Building for NVIDIA (HIP on CUDA)..."
        build_target "nvidia" \
            -DCMAKE_PREFIX_PATH="$HIP_PATH" \
            -DWITH_HIP=ON \
            -DHIP_PLATFORM=nvidia \
            || echo "Failed to build for NVIDIA."
        ;;
    (cpu)
        echo "Building CPU-only..."
        build_target "cpu" \
            -DWITH_HIP=OFF \
            || echo "Failed to build for CPU-only."
        ;;
    (orochi)
        echo "Building Orochi (unified GPU via runtime dispatch)..."
        build_target "orochi" \
            -DWITH_OROCHI=ON \
            -DWITH_HIP=OFF \
            || echo "Failed to build for Orochi."
        ;;
    (all)
        echo "Building for AMD (ROCm)..."
        build_target "amd" \
            -DCMAKE_PREFIX_PATH="$HIP_PATH" \
            -DWITH_HIP=ON \
            -DHIP_PLATFORM=amd \
            || echo "Failed to build for AMD."

        echo "Building for NVIDIA (HIP on CUDA)..."
        build_target "nvidia" \
            -DCMAKE_PREFIX_PATH="$HIP_PATH" \
            -DWITH_HIP=ON \
            -DHIP_PLATFORM=nvidia \
            || echo "Failed to build for NVIDIA."

        echo "Building CPU-only..."
        build_target "cpu" \
            -DWITH_HIP=OFF \
            || echo "Failed to build for CPU-only."
        ;;
    (*)
        echo "Invalid target: '$TARGET'"
        echo "Usage: $0 <TNN_VERSION> [amd|nvidia|cpu|orochi|all]"
        exit 1
        ;;
esac
