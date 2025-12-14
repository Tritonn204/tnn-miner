# HIPRTC Isolation Test

## Purpose

This test determines whether HIPRTC crashes are **intrinsic to HIPRTC itself** or caused by **external interference** from other parts of the program.

## What It Does

The test runs **very early** in `main()` via `GPUTest()`, before:
- Thread pools are created
- GPU mining kernels are loaded
- Network connections are established
- Most other initialization occurs

It attempts to compile a minimal HIP kernel using HIPRTC with full SEH exception handling on Windows.

## Location

- Test code: `src/tnn_hip/core/test_hiprtc_isolation.hpp`
- Called from: `src/core/gpulibs.h::GPUTest()`
- Runs at: `src/core/miner.cpp:329` (tnn_main)

## Interpreting Results

### If the Test PASSES (no crash)

```
========================================
[ISOLATION TEST] SUCCESS!
[ISOLATION TEST] HIPRTC compilation works in isolation
========================================
```

**This means:**
- HIPRTC itself works correctly
- Any crashes occurring later are from:
  - Thread interactions
  - Memory corruption from other code
  - Race conditions with GPU state
  - Concurrent access to HIPRTC from multiple threads

**Action:** Look for:
- Multiple threads calling HIPRTC simultaneously
- Memory corruption bugs in mining kernels
- Heap corruption
- Race conditions

### If the Test FAILS (crashes)

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[ISOLATION TEST] CRASH DETECTED in hiprtcCreateProgram!
[ISOLATION TEST] Exception code: 0xC0000005
[ISOLATION TEST] Exception type: EXCEPTION_ACCESS_VIOLATION
[ISOLATION TEST] Context: Running in ISOLATED environment on main thread
[ISOLATION TEST] This confirms the crash is FROM hiprtc itself, NOT interference
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

**This means:**
- HIPRTC has an intrinsic bug or environmental issue
- The problem is NOT from your code interfering

**Action:** Check:
- HIPRTC DLL/library versions
- GPU driver compatibility
- Missing dependencies
- Corrupted HIPRTC installation
- Known HIPRTC bugs for your GPU/driver version

## Exception Codes (Windows)

Common codes you might see:

- `0xC0000005` - EXCEPTION_ACCESS_VIOLATION (illegal memory access)
- `0xC00000FD` - EXCEPTION_STACK_OVERFLOW
- `0xC000008C` - EXCEPTION_ARRAY_BOUNDS_EXCEEDED
- `0xC000001D` - EXCEPTION_ILLEGAL_INSTRUCTION

## Key Insight

The SEH `__try/__except` blocks in the original code **cannot distinguish** between:
1. Crashes from HIPRTC itself
2. Crashes from memory corruption that manifests during HIPRTC calls

This isolation test **eliminates option 2** by running before any complex state exists.

## Technical Details

### Why This Works

The test runs when:
- Only one thread exists (main)
- No GPU kernels loaded yet
- Minimal heap activity
- No network sockets
- Clean memory state

If it crashes here, external interference is impossible.

### What Gets Tested

1. `hipGetDeviceCount()` - Basic HIP runtime
2. `hiprtcCreateProgram()` - Program object creation
3. `hiprtcCompileProgram()` - Actual compilation
4. Code retrieval and cleanup

Each step has SEH protection and detailed logging.

## Example Output

### Success Case
```
========================================
[ISOLATION TEST] Starting HIPRTC isolation test
[ISOLATION TEST] This runs BEFORE any other GPU initialization
[ISOLATION TEST] Thread ID: 12345
[ISOLATION TEST] Process ID: 67890
========================================
[ISOLATION TEST] Step 1: Checking HIP runtime initialization
[ISOLATION TEST] hipGetDeviceCount returned: 1 (error code: 0)
[ISOLATION TEST] Found 1 HIP device(s)
[ISOLATION TEST] Step 2: Creating HIPRTC program
[ISOLATION TEST] Kernel source length: 234 bytes
[ISOLATION TEST] Step 2a: Calling hiprtcCreateProgram (with SEH protection)...
[ISOLATION TEST] Step 2b: hiprtcCreateProgram returned successfully
[ISOLATION TEST] Return code: 0
[ISOLATION TEST] Step 3: Compiling HIPRTC program
[ISOLATION TEST] Step 3a: Calling hiprtcCompileProgram (with SEH protection)...
[ISOLATION TEST] Step 3b: hiprtcCompileProgram returned successfully
[ISOLATION TEST] Return code: 0
[ISOLATION TEST] Step 4: Retrieving compiled code
[ISOLATION TEST] Compiled code size: 4567 bytes
[ISOLATION TEST] Step 5: Cleaning up

========================================
[ISOLATION TEST] SUCCESS!
[ISOLATION TEST] HIPRTC compilation works in isolation
[ISOLATION TEST] If crashes occur later, they are from:
[ISOLATION TEST]   - Thread interactions
[ISOLATION TEST]   - Memory corruption from other code
[ISOLATION TEST]   - Race conditions with GPU state
========================================
```

### Failure Case
```
========================================
[ISOLATION TEST] Starting HIPRTC isolation test
...
[ISOLATION TEST] Step 2a: Calling hiprtcCreateProgram (with SEH protection)...

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[ISOLATION TEST] CRASH DETECTED in hiprtcCreateProgram!
[ISOLATION TEST] Exception code: 0xC0000005
[ISOLATION TEST] Exception type: EXCEPTION_ACCESS_VIOLATION (memory access error)
[ISOLATION TEST] Context: Running in ISOLATED environment on main thread
[ISOLATION TEST] This confirms the crash is FROM hiprtc itself, NOT interference
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Error: hiprtcCreateProgram crashed in isolation test
```
