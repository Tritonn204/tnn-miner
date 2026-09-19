# RDNA4 Pearl adapter

Automatically selected for gfx1200 and gfx1201. Mining, benchmark and shape
tuning require a successful process-local CPU/proof qualification on the
selected device. Compilation alone is not hardware qualification.

Iris owns the RDNA4 WMMA and tiled GEMM implementation. This directory owns
Pearl's native 4x32 ticket mapping, rank-128 checkpoints, transcript storage,
keyed BLAKE3 and winner collection. The proof configuration is unchanged.

Production uses recipe 0. Other K-step, LDS-bank and load-width combinations
remain compile-time study variants, not public CLI switches or production
tuning choices. Shape tuning uses the existing Pearl tuner: up to 32 shapes,
batch 16, M/N on a 1024 grid through 16384 and K=2048/4096/8192.

Use normal device selection and the same commands as other Pearl backends:

```powershell
.\tnn-miner.exe --hip-test-pearl --no-cpu
.\tnn-miner.exe --bench-pearl --no-cpu
.\tnn-miner.exe --tune-pearl --no-cpu
```

Benchmarking validates captured CPU proofs at the requested shape before
timing fresh preparation, fused jackpots and readback. Diagnostic D/transcript
writes are absent from timed entry points. Full mining uses the normal --prl
command and pool settings. No architecture override or AOT-loading hook exists.
