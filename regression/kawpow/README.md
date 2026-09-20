# KawPow reduction regression

Run `python regression/kawpow/check_reduction.py` from the repository root.
Requires Python and `clang++`; no GPU or HIP SDK is needed for the host test.
The test extracts the actual production reducers, not a duplicate implementation.

Explicit hardware check: add `--gpu` on a Windows gfx1100 development machine
with ROCm 6.4. This performs 48 tiny launches with a 20-second process timeout,
no DAG allocation and no mining. Do not run alongside a miner.

## Ownership contracts

- Monolithic, split single-hash and global-L1 kernels use `progpow_reduce_full`:
  every lane participates in the gather before lane 0 reads all eight words.
- Two-way, seed64/digest and four-way kernels use distributed reducers:
  lanes 0–7 each own one word and perform cooperative stores.
- Logical groups are always 16 lanes, independent of wave32/wave64/warp32.
  NVIDIA shuffle masks cover only that group, including partial N-way batches.

Host tests cover wave32/wave64 grouping and untouched private slots. The GPU
fixture checks one-, two- and four-way batches of sizes 1–16, compares full and
distributed digests with scalar FNV references, and checks output guards.

These are reduction tests, not full proof or pool validation. NVRTC compilation
does not replace real NVIDIA execution; retest there before claiming full
cross-vendor validation.
