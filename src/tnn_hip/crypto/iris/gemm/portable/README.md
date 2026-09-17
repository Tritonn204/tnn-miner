# Correctness-first SIMT Pearl fallback

Use the new backend-neutral `../dispatch.hpp` interface. It exposes the same
materialized signed-int8 GEMM, full int32 D, rank128 checkpoints, and winner ABI
as the qualified recipe. It does not generate noise or construct proofs.

```cpp
namespace gemm = tnn::hip::iris::gemm;

gemm::Problem problem{
    a, b, d, m, n, k, lda, ldb, ldd, 128,
    key, target, results, diagnostic
};

auto status = gemm::launch(
    problem,
    gemm::Mode::Fused,
    stream,
    gemm::Backend::Automatic
);
```

## Selection and build integration

- Exact `gfx1100` selects the qualified recipe when its object is linked.
- Other HIP targets select portable SIMT; gfx1101 does not inherit qualification.
- `Backend::PortableSimt` forces fallback, including on gfx1100.
- CDNA MFMA is experimental and is never an automatic selection.
- A build without the qualified object falls back on gfx1100 for Automatic;
  an explicit request for the missing qualified backend returns an error.

Compile `dispatch.cpp` and `portable/kernel.hip.cpp` for the intended target.
Define `IRIS_DISPATCH_WITH_GFX1100` for the dispatcher only when also linking the
qualified gfx1100 object. Build that object separately for gfx1100: its deliberate
architecture guard must not be disabled to make a multi-target build pass.
The legacy WMMA-specific `DefaultBackend` and existing TNN calls are unchanged;
they require migration to this interface during TNN E2E integration.

## Recipe and safety contract

`SimtPearl` combines `PearlTicketLayout` with `ConservativeSchedule`. Logical
thread ownership is independent of hardware wave width. The factors of 16/32 in
the ticket formulas describe Pearl's qualified logical grouping, not a wave
shuffle or hardware lane assumption. One owner holds the cumulative 128 output
values of one ticket, folds them at rank boundaries, and hashes/checks at the end.

Input and mode validation reuse the qualified host contract. Matrix dimensions
and padding remain restricted to that contract, not arbitrary tail shapes.
Only regular integer operations, cooperative loads and block barriers are used.
The five compiled target assemblies contain no WMMA/MFMA instructions.

Each launch covers at most eight complete 128x256 output tiles, with all K ranks
inside the launch. `launch_portable_range` permits explicit per-range timing and
cancellation. `launch` enqueues all ranges in stream order; it does not synchronize
or enforce a time budget itself. Callers must reset state once before the entire
problem, never between ranges. They retain ownership of memory and stream
lifetimes and must handle partial completion if a later enqueue fails.

This is not a throughput-qualified implementation. The SIMT fallback spills
heavily on compiled CDNA targets; use the resource audit rather than assuming
register behavior transfers from RDNA. Only gfx1100 has small-fixture GPU
correctness evidence so far. Successful cross-compilation is not hardware
qualification, and a small test is not full-size performance validation.

See `gemm_next/iris_portable/RESULTS.md` for exact evidence and limits.
