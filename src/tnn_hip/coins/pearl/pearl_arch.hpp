#pragma once
#include <string_view>

namespace tnn::pearl {

inline constexpr std::string_view architecture_name(std::string_view name) {
    return name.substr(0, name.find(':'));
}

inline constexpr bool cdna_target(std::string_view name) {
    name = architecture_name(name);
    return name == "gfx908" || name == "gfx90a" || name == "gfx942";
}

inline constexpr bool rdna3_target(std::string_view name) {
    name = architecture_name(name);
    return name == "gfx1100" || name == "gfx1101" || name == "gfx1102";
}

inline constexpr bool rdna4_target(std::string_view name) {
    name = architecture_name(name);
    return name == "gfx1200" || name == "gfx1201";
}

// Exact discrete-GPU inventory. A numeric range is not an ISA capability test.
inline constexpr bool portable_target(int gfx) {
    switch (gfx) {
    case 900: case 906:
    case 1010: case 1011: case 1012:
    case 1030: case 1031: case 1032: case 1033: case 1034:
    case 1100: // Local execution control, not a replacement for qualified WMMA.
        return true;
    default:
        return false;
    }
}

// Native signed dot4 is absent on original Vega and RDNA1. gfx1100 is
// included only as the local control, using its distinct mixed-sign opcode.
inline constexpr bool portable_dot_target(int gfx) {
    switch (gfx) {
    case 906:
    case 1030: case 1031: case 1032: case 1033: case 1034:
    case 1100:
        return true;
    default:
        return false;
    }
}

inline constexpr bool legacy_mining_target(int gfx) {
    return gfx != 1100 && portable_target(gfx);
}

inline constexpr unsigned portable_recipe(int gfx) {
    return portable_dot_target(gfx) ? 2u : 0u;
}

} // namespace tnn::pearl
