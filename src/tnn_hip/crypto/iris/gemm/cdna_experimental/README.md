# Early CDNA2 MFMA path — compile-only

This gfx90a-only raw GEMM prototype uses signed-i8
`v_mfma_i32_16x16x16i8`, one wave64 per 16x16 output tile. It has no checked public
launcher, no noise generation, no jackpot epilogue and no automatic dispatch.
Its entrypoint assumes full tiles, positive K divisible by 16, valid strides
and allocations, and exactly 64 threads. Do not call it as a qualified backend.

## Source and mapping evidence

The instruction and register mappings come from AMD's
[matrix-core guide](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
and [matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator).
All 768 A/B/D elements were checked against the installed calculator; its source
hash and complete layout output are retained in the build evidence.

For lane `l`, input byte `b`, and accumulator register `r`:

- A: row `l % 16`, K coordinate `4 * (l / 16) + b`.
- B: K coordinate `4 * (l / 16) + b`, column `l % 16`.
- D: row `4 * (l / 16) + r`, column `l % 16`.

The CPU model verifies signed GEMM reconstruction from these fragments and the
coverage of the future Pearl adapter. Unlike the qualified RDNA3 implementation,
one CDNA lane's four accumulators cover two row-pair tickets, and a complete
ticket spans multiple lanes and nonadjacent N tiles.

For a global output `(row, col)`, the Pearl ticket row is `row & ~1u` and its
column representative is `(col / 256) * 256 + ((col % 32) / 16) * 16 + col % 2`.
Each ticket collects exactly 128 accumulator values in the CPU ownership test.
A future fused implementation must combine those values at each cumulative
rank boundary before transcript updates. Dropping in the RDNA3 per-thread
checkpoint fragment would be incorrect.

## Status

The prototype emits the intended MFMA instruction, uses zero scratch/LDS, and
has a 40-byte kernel argument segment. Resource comments distinguish 18 regular
VGPRs from 24 total vector-register accounting; these are not two kernels.

There is no CDNA GPU execution evidence. No performance claim, fused-Pearl
qualification, or instruction parity with RDNA is implied. This prototype and
the ownership model are a starting point for a later CDNA hardware pass, not a
prerequisite for gfx1100 TNN E2E.
