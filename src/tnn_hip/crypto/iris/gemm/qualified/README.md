# Qualified gfx1100 Pearl recipe

This is the maintained specialization used by TNN's integrated gfx1100 Pearl miner,
HIP validation and benchmark. It does not replace `DefaultBackend`. Historical `../experimental` sources and
`../../pearl` checkpoint experiments remain frozen evidence, not the public API
for this recipe.

## Start here

```cpp
#include "recipe.hpp"
#include "api.hpp"

namespace qualified = tnn::hip::iris::gemm::qualified;

using MyRecipe = qualified::Recipe<
    qualified::Gfx1100,
    qualified::Paired128x256,
    qualified::PearlRank128>;

static_assert(std::is_same_v<MyRecipe, qualified::Gfx1100Pearl>);
```

`kernel.hip.cpp` contains one kernel template parameterized by this recipe and
instantiates raw, fused, and diagnostic modes. `launch(...)` exposes the named
recipe through a normal HIP host interface. Compile that translation unit for
gfx1100 and link it with the caller; including the header alone does not build
or register kernels. All three modes write full D.

| Struct | Responsibility | Qualified choices |
|---|---|---|
| `Gfx1100` | Instruction and lane/layout mechanics | Wave32, signed i8 WMMA, paired LDS reads, padded B, local barriers |
| `Paired128x256` | Tile and scheduling | 128x256x32, 256 threads, early prefetch, shared address rematerialization, non-streamed operands, mapping 1 |
| `PearlRank128` | Checkpoint policy | Rank 128, four XOR chains, predicated transcript updates |

`prefetch` chooses placement, not enablement: 0 is before operand reads, 1 is
between operand reads and WMMA, 2 is after checkpointing. `prefetch_enabled`
controls enablement separately. `shared_address` rematerializes addresses once
for both operand halves; `rematerialize` is the alternative per-operand path
when shared addressing is disabled. The qualified recipe uses shared addressing.

The parameters are compile-time source structure, **not an automatic tuner or
portability promise**. Only the named struct combination is accepted by
`Recipe`. Extending the whitelist requires a separately qualified implementation.
Fixed WMMA operand order (B before A), packed register permutations, instruction
immediates and rank transcript dimensions are architecture/algorithm mechanics,
not generic knobs. Disabled historical instruction branches in `ops.hpp` and
`checkpoint.inc` are not exposed as qualified configurations.

## Public launch contract

Use `Mode::Raw`, `Mode::Fused`, or `Mode::Diagnostic`. There is no runtime streamed
flag and no overlapping raw/force-full/diagnostic booleans. Diagnostic mode
requires a non-null aligned diagnostic output. Raw and fused modes do not write
that output, even if a pointer is supplied.

Inputs retain the original contract: materialized/noised signed-i8 operands,
int32 full output, M/N up to 8192 with 128/256 tile divisibility, K=2048 or 4096,
rank 128, and validated alignment/strides/padding. This API does not generate
noise. The caller owns buffer sizes, stream lifetime, and state reset.

`NativeWinner` is 40 bytes, `NativeState` 8 bytes, and `NativeResults` 24 bytes on
the qualified 64-bit target. Winners are appended atomically after hashing and
target comparison. Capacity exhaustion sets overflow; capacity zero remains a
supported benchmark setting. Raw mode retains the historical pointer contract
even though the kernel does not consume all Pearl metadata.

## Why the checkpoint is an include fragment

`checkpoint.inc` is deliberately included inside the K loop after both WMMA
halves, before the end-of-iteration barrier. It requires `c`, `transcript`, `kk`,
and the `Pearl` policy alias. Accumulators stay cumulative across rank boundaries.
There is no include guard because this is a lexical fragment, not a header API.

Force-inlining a function does not guarantee the same compiler intermediate
representation as writing statements in the loop. In controlled probes, the
function-boundary form used 251 diagnostic VGPRs rather than 240 and changed
instructions. Moving only reduction into a function matched resources but still
failed exact ISA comparison. Placing the body back in the mainloop restored
byte-for-byte parity. This demonstrates a source-boundary effect with this
compiler; it does not identify the particular compiler pass responsible.

The successful strategy was to parameterize known-good mechanics while
preserving sensitive expression structure and statement placement. Whitespace,
clear declarations, and comments are compatible with this strategy. Do not
replace the fragment with a callback merely for stylistic uniformity.

## Qualification

The historical qualification harness is archived locally under
`study/gemm_next/iris_recipe/qualify.py`; it is not a shipping build dependency.
The script compiles and links but launches **no GPU kernels**. Only a pure host
validation executable is run; the HIP-linked launcher test is never executed.

Acceptance requires exact retained instruction bytes for all three modes,
identical code-object metadata except symbol names, unchanged occupancy and
register counts, input/mode tests, and compile-time rejection of unsupported
recipes. New artifacts go into timestamped folders; historical evidence is not
overwritten. A changed symbol/module hash is expected after namespace/template
changes and must not be substituted into the old TNN identity header.

The historical ISA comparison qualifies the host wrapper's kernel bodies, not
the full mining pipeline. The integrated production path is checked separately
by `--hip-test-pearl`; `--bench-pearl` measures that path offline without network
or proof submission.
