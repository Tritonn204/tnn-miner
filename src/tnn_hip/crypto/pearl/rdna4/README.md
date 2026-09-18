# Experimental gfx1201 Pearl adapter

This is an offline tester backend, **not a qualified mining backend**. Normal
mining must reject it until device correctness and accepted proofs are recorded.

Iris owns the RDNA4 WMMA and tiled GEMM implementation. This directory owns
Pearl's native 4x32 ticket mapping, rank-128 checkpoints, transcript storage,
keyed BLAKE3 and winner collection. The proof configuration is unchanged.

The tester's recipe index selects K-step 32/64, one/two LDS banks, and 8/16-byte
input loads. It is deliberately not a generic production tuning-cache entry.
Compile/CPU checks do not establish performance or GPU synchronization safety.

Offline CLI (gfx1201 as device 0):

```powershell
.\tnn-miner.exe --hip-test-pearl-gfx12 --pearl-test-recipe 0
.\tnn-miner.exe --bench-pearl-gfx12 --pearl-test-recipe 0 --pearl-test-workload 0
```

Only benchmark after correctness passes. Workloads are 0=fresh prep+fused,
1=prepared fused, 2=prepared D-free raw. Prepared controls cannot be used in
mining mode. All benchmark shapes verify captured proofs before measurement.
Diagnostic D/transcript writes are absent from the timed entry points.
