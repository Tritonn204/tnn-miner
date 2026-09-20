#pragma once

#include "coordinate.hpp"

namespace iris::hip {

enum class AddressSpace {
    Global,
    Lds,
    Generic,
};

template <typename T, AddressSpace Space>
struct BufferView {
    using data_type = T;
    static constexpr AddressSpace address_space = Space;

    T* ptr;
    int size;

    IRIS_HOST_DEVICE_INLINE bool in_bounds(int offset) const {
        return offset >= 0 && offset < size;
    }

    IRIS_HOST_DEVICE_INLINE T* ptr_at(int offset) const {
        return ptr + offset;
    }

    IRIS_DEVICE_INLINE T& ref(int offset) const {
        return ptr[offset];
    }

    IRIS_DEVICE_INLINE T load(int offset) const {
        return ptr[offset];
    }

    IRIS_DEVICE_INLINE void store(int offset, T v) const {
        ptr[offset] = v;
    }
};

template <typename T>
using GlobalBuffer = BufferView<T, AddressSpace::Global>;

template <typename T>
using LdsBuffer = BufferView<T, AddressSpace::Lds>;

template <typename T>
using GenericBuffer = BufferView<T, AddressSpace::Generic>;

template <typename T>
IRIS_HOST_DEVICE_INLINE GlobalBuffer<T> make_global_buffer(T* ptr, int size) {
    return {ptr, size};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE LdsBuffer<T> make_lds_buffer(T* ptr, int size) {
    return {ptr, size};
}

template <typename T>
IRIS_HOST_DEVICE_INLINE GenericBuffer<T> make_generic_buffer(T* ptr, int size) {
    return {ptr, size};
}

} // namespace iris::hip
