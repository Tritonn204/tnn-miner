# Pearl native128 integration

The production gfx1100 Pearl kernel uses a 128x128x32 tile, four wave32 waves,
two LDS staging banks, and native 4x32 candidate ownership. There is one mining
backend; raw and diagnostic entry points are validation tools, not fallbacks.

## Read the implementation

- Iris `gemm/native128/recipe.hpp`: architecture ownership, padding, and schedule traits.
- `layout.hpp`: candidate coordinates and checkpoint transcript placement.
- `rtc.hip`: tile mapping, GEMM, complete BLAKE3, target comparison, winners.
- Iris `gemm/native128/asm_templates.hpp`: string-only WMMA templates with fixed operand order.
- `slots.hpp`: fused schedule, including rank-128 checkpoint reductions.
- Iris `gemm/native128/raw_slots.hpp`: matching raw schedule without checkpoints or output stores.

The fixed-register schedule is a qualified specialization, not an arbitrary
shape generator. Changing traits alone cannot retarget its physical registers.
Keep each schedule in one volatile assembly block: splitting it lets the compiler
allocate across previously protected register lifetimes.

## Contracts that must survive edits

- Fused: 253 VGPRs, 34 SGPRs, 25,600 bytes LDS, **zero scratch/spills** on the
  qualified ROCm 6.4 gfx1100 build. Check the entire emitted body, including
  internal assembly labels, not just the WMMA blocks.
- Each lane owns four rows and 32 columns. Configuration bytes, job key, noise
  seeds, CPU proof ownership and GPU ownership must describe that same layout.
- The jackpot work multiplier is `128 * K`; it is not the mining-loop batch size.
- Every attempt gets fresh A, its own seed, counters and winner region. B and the
  zero-A Merkle tree are job-stable, separately cached for user and dev channels.
- Mining queues 16 GEMMs and waits once per batch. It does not store D or
  diagnostic transcripts. Empty attempts require no proof-data downloads.
- Winning evidence is copied before buffers are reused. Overflow fails visibly;
  it must never silently discard candidates.

## Production checks

From the built miner directory:

```powershell
.\tnn-miner.exe --hip-test-pearl
.\tnn-miner.exe --bench-pearl --bench-pearl-seconds 120
```

The test checks CPU/GPU candidate coverage, captured proofs, guards, overflow,
certificate versions 2/3, all three supported K values, and a rectangular Merkle tree.
The benchmark uses the production batch callbacks with fresh preparation and
readback. Its wall rate excludes initial job preparation, network and proof
submission; live-pool rates include those effects. Run GPU checks serially.

## Shape tuning

Mining uses a validated cached shape, or performs an offline startup sweep when
no usable cache exists. `--tune-pearl` runs that selection without connecting to
a pool; `--gpu-retune` forces a new sweep. `--gpu-no-tune` bypasses both timed
tuning and the cache, using 8192x8192x4096 with 16 GEMMs per batch.

The tuner tests at most 32 distinct shapes: 24 coarse candidates spanning M/N
through 16384 and K=2048/4096/8192, then eight local refinements on the 1024 grid.
Every candidate must pass captured-winner CPU/proof checks before timing.
Timing includes fresh A preparation, fused GEMMs and readback, but excludes
initial job setup and proof submission. The default and two screening leaders
receive three alternating confirmation rounds. A replacement must improve the
median by at least 1% and beat the default in at least two rounds.

Iris owns computational shape limits; Pearl intersects them with its proof
contract. This is a bounded adaptive search, not an exhaustive Cartesian sweep.
Cache validation checks backend/version, dimensions, launch geometry, batch,
measurement fields and available memory. Only a completed, applied tune is saved.
No additional architecture is qualified by this tuner.
