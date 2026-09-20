#pragma once

// Test-only definitions of the miner's globals. The API implementation, rate
// definitions and device filtering are the production sources, not mocks.
#include <algo_definitions.h>
#include <atomic>
#include <set>

struct FixtureProfile {
    struct Coin { int miningAlgo = ALGO_XELISV3; } coin;
};
inline FixtureProfile miningProfile;
inline constexpr int DEVICE_SHARE_CPU = 32;
inline std::atomic<int> deviceAccepted[33]{}, deviceRejected[33]{};
inline std::set<int> HIP_includeDevices, HIP_excludeDevices;
