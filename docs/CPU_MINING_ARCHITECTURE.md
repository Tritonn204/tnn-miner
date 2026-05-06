# CPU Mining Architecture

## Overview

The CPU mining system uses a unified, object-oriented architecture that provides a consistent interface for all CPU mining algorithms. This replaces the previous procedural approach with a cleaner, more maintainable design.

## Architecture Components

### 1. ICPUAlgorithm Interface (`src/tnn_cpu/common/cpu_algo.hpp`)

The base interface that all CPU algorithms must implement:

```cpp
class ICPUAlgorithm {
    virtual bool initialize(int thread_id) = 0;
    virtual void cleanup() = 0;
    virtual bool set_work(const uint8_t* work_template, size_t size) = 0;
    virtual bool compute_hash(uint64_t nonce, uint8_t* output) = 0;
    virtual const CPUAlgoConfig& get_config() const = 0;
};
```

**Key Methods:**
- `initialize()` - Allocates worker data, huge pages, etc. Called once per thread
- `set_work()` - Updates work template, performs preprocessing (e.g., matrix computation)
- `compute_hash()` - Computes a single hash for a given nonce
- `get_config()` - Returns algorithm metadata (template size, nonce offset, algo ID, etc.)

### 2. CPUMiner Class (`src/tnn_cpu/common/cpu_miner.hpp`)

Lightweight wrapper that manages the mining loop:

```cpp
class CPUMiner {
    bool mine_one(uint8_t* hash_output, uint64_t* found_nonce, uint8_t* work_output);
    void set_work(const uint8_t* work_template, size_t size);
    void set_difficulty(uint64_t difficulty);
};
```

**Features:**
- No internal threading (mining thread calls `mine_one()` directly)
- Automatic nonce encoding with thread ID
- Integrated difficulty checking using existing `CheckHash()`
- Hash counter tracking

### 3. CPUAlgoRegistry (`src/tnn_cpu/common/cpu_algo_registry.hpp`)

Singleton factory for creating algorithm instances:

```cpp
auto miner = CPUAlgoRegistry::instance().create("xelis_v2");
```

Algorithms self-register using the `REGISTER_CPU_ALGORITHM` macro:

```cpp
REGISTER_CPU_ALGORITHM("xelis_v2", XelisV2CPU)
```

### 4. Unified Mining Function (`src/tnn_cpu/common/mine_cpu_unified.hpp`)

Single mining function that works with any registered algorithm:

```cpp
void mineCPU_unified(int tid, const std::string& algo_name);
```

**Handles all boilerplate:**
- Job synchronization (height tracking, jobCounter)
- Dev mining probability switching
- Protocol-specific work parsing and share building
- Share submission coordination
- Connection handling
- Counter batching

## Migrated Algorithms

### Currently Migrated

| Algorithm | Class Name | File | Status |
|-----------|------------|------|--------|
| Xelis V2 | `XelisV2CPU` | `src/tnn_cpu/algos/xelis_cpu.hpp` | ✅ Tested |
| Xelis V3 | `XelisV3CPU` | `src/tnn_cpu/algos/xelis_cpu.hpp` | ✅ Tested |
| Astrix | `AstrixCPU` | `src/tnn_cpu/algos/astrix_cpu.hpp` | ✅ Ready |
| Nexellia | `NexelliaCPU` | `src/tnn_cpu/algos/nexellia_cpu.hpp` | ✅ Ready |
| Waglayla | `WaglaylaCPU` | `src/tnn_cpu/algos/waglayla_cpu.hpp` | ✅ Ready |
| Hoosat | `HoosatCPU` | `src/tnn_cpu/algos/hoosat_cpu.hpp` | ✅ Ready |

### Pending Migration

- Dero (AstroBWT)
- Spectre
- RandomX0 variants
- Verus
- Shai
- Yespower
- Rinhash

## Algorithm Implementation Guide

### Step 1: Create Algorithm Class

```cpp
class MyAlgoCPU : public ICPUAlgorithm {
public:
    MyAlgoCPU() : worker_(nullptr), thread_id_(0) {}

    bool initialize(int thread_id) override {
        worker_ = allocate_worker();  // Your allocation logic
        return worker_ != nullptr;
    }

    void cleanup() override {
        if (worker_) {
            free(worker_);
            worker_ = nullptr;
        }
    }

    bool set_work(const uint8_t* work_template, size_t size) override {
        std::memcpy(work_buffer_, work_template, size);
        // Optional: preprocessing like matrix computation
        return true;
    }

    bool compute_hash(uint64_t nonce, uint8_t* output) override {
        uint8_t local_work[TEMPLATE_SIZE];
        std::memcpy(local_work, work_buffer_, TEMPLATE_SIZE);
        std::memcpy(local_work + NONCE_OFFSET, &nonce, 8);

        my_hash_function(local_work, *worker_, output);
        return true;
    }

    const CPUAlgoConfig& get_config() const override {
        static const CPUAlgoConfig config = {
            .name = "myalgo",
            .template_size = TEMPLATE_SIZE,
            .hash_size = 32,
            .nonce_offset = NONCE_OFFSET,
            .nonce_size = 8,
            .needs_hugepages = true,
            .needs_preprocessing = false,
            .algo_id = ALGO_MYALGO
        };
        return config;
    }

private:
    Worker* worker_;
    int thread_id_;
    uint8_t work_buffer_[TEMPLATE_SIZE];
};

REGISTER_CPU_ALGORITHM("myalgo", MyAlgoCPU)
```

### Step 2: Create Wrapper Function

```cpp
// src/tnn_cpu/mine_myalgo_unified.cpp
#include "../coins/miners.hpp"
#include "common/mine_cpu_unified.hpp"
#include "algos/myalgo_cpu.hpp"

void mineMyAlgo_unified(int tid) {
    mineCPU_unified(tid, "myalgo");
}
```

### Step 3: Update Build System

```cmake
# cmake/myalgo.cmake
list(APPEND myAlgoSources
    src/tnn_cpu/mine_myalgo_unified.cpp
)
```

### Step 4: Update Routing

```cpp
// src/coins/miners.hpp
void mineMyAlgo_unified(int tid);

// In getMiningFunc():
case ALGO_MYALGO:
    return mineMyAlgo_unified;
```

## Protocol Support

### Xelis Protocols
- **PROTO_XELIS_SOLO** - Job field: `miner_work`, hex parsing
- **PROTO_XELIS_XATUM** - Job field: `miner_work`, base64 parsing
- **PROTO_XELIS_STRATUM** - Job field: `miner_work`, hex parsing

### KAS-family Protocols
- **PROTO_KAS_SOLO** - Job field: `template`, hex parsing
- **PROTO_KAS_STRATUM** - Job field: `template`, hex parsing, nonce as decimal→hex

The unified function automatically handles protocol differences for:
- Work template parsing
- Share submission formatting
- Job field names

## Key Design Decisions

### 1. No Internal Threading in CPUMiner
- Keeps it simple and matches existing architecture
- Mining thread calls `mine_one()` directly in a loop
- Avoids thread synchronization overhead

### 2. Dev Mining at Function Level
- Probabilistic switching between two CPUMiner instances
- Each instance maintains its own work and difficulty
- Cleaner than switching within the algorithm

### 3. Reuse Existing Infrastructure
- `CheckHash()` for difficulty validation
- `ConvertDifficultyToBig()` for target conversion
- Global variables from `net.hpp` for job coordination
- Terminal colors and utilities

### 4. Registry Pattern
- Same as GPU side for consistency
- Easy to add new algorithms
- Automatic registration via macro

## Benefits

✅ **Unified Interface** - All algorithms implement same interface
✅ **Less Boilerplate** - Algorithms only implement hashing logic
✅ **Easier Testing** - Can test algorithms in isolation
✅ **Future-Proof** - Easy to add new algorithms
✅ **Consistent** - Matches GPU architecture pattern
✅ **Non-Breaking** - Legacy algorithms still work

## Non-Breaking Migration

The new system is **completely non-breaking**:

- Only migrated algorithms use new system
- Legacy algorithms continue using old `src/coins/mine_*.cpp` files
- CMake routes each algorithm individually
- Can migrate incrementally, algorithm by algorithm
- Old and new systems coexist peacefully

## Next Steps

### For CPU Migration
1. Continue migrating simpler algorithms (Spectre, Shai, Rinhash, Yespower)
2. Then tackle complex ones (Dero, RandomX0, Verus)

### For GPU Migration
Use this same architecture pattern:
- `IGPUAlgorithm` interface (already exists)
- `GPUMiner` wrapper (already exists)
- Move to RTC (Runtime Compilation) for kernels
- Protocol-aware unified GPU mining function
