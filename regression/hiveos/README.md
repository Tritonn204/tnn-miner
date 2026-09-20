# HiveOS and mmpOS release integration tests

This independent compatibility harness installs and launches the **candidate
release archive**, not a replacement test miner. Nothing here is linked into
TNN or included in the release archive. `harness.py` contains generic helpers;
`tnn.py`, `mmpos.py`, the scenarios and the C++ fixture are the TNN adapters. This boundary
allows a future standalone repository without an external service dependency.

## Run

From the repository root, with a Linux Docker engine:

```sh
bash regression/hiveos/ci.sh \
  tnn-miner-v0.9.0.hiveos_mmpos.amd64.tar.gz \
  Tnn-miner-amd64-v0.9.0.tar.gz

# Same artifacts, separate mmpOS release gate:
bash regression/hiveos/ci.sh \
  tnn-miner-v0.9.0.hiveos_mmpos.amd64.tar.gz \
  Tnn-miner-amd64-v0.9.0.tar.gz build/mmpos-results mmpos
```

Use the actual two artifacts from the same pipeline. Filenames vary by version;
CI passes their exported names. The image builds a separate production-API
fixture, installs dependencies, then runs with **no network except loopback**,
no GPUs, no host mounts, all capabilities dropped, 2 CPUs and 2 GiB RAM.
The test container is disposable; never run `run.py` directly on a Hive rig.

Outputs are under `build/hiveos-results`: JUnit, report JSON, archive/binary
hashes, container image/package receipts, production API and Hive stats snapshots, flight-sheet/launch logs and
the local Stratum transcript. The scripts return nonzero on failure. CI makes
both this test and the existing shellcheck/structure job release dependencies.
Run duration is bounded to 30 mining seconds, 90 seconds including startup.
The local server accepts submissions except for one scripted rejection. It
tests protocol/reporting, **not proof validity**. No real payout address is used.

## Coverage and boundaries

- Package: local HTTP installation URL, archive paths, permissions, required
  callbacks, binary identity and bundled-library/tune identity.
- Flight sheet: existing TNN extra-argument configuration, wrapper launch,
  config paths, API enablement, failure propagation and cleanup.
- Production API fixture: CPU/GPU/hybrid, selected/excluded/reordered devices,
  zero-rate startup, missing telemetry, large Pearl MAC/s values.
- Stats callback: sourced as the Hive agent sources it; exported `khs` and
  `stats`, physical-unit metadata, PCI association, shares, version and uptime.
- API failure: malformed/unavailable responses clear stale samples.
- Live package: one-thread local Xelis v3 CPU mining, no GPU/MSR/affinity
  changes, actual HTTP API and accepted/rejected response accounting.
- Negative controls: missing callbacks, wrong binary digest, 1000x unit error,
  swapped telemetry. A validator that accepts a negative control fails the job.

GPU cases are **reporting-contract simulations**, not GPU mining or a Hive
cloud/dashboard test. Windows ZIP execution and real GPU validation are not
covered. `hs_units=hs` is Hive's scale selector; Pearl
numbers remain MAC/s, never converted to an invented effective H/s. Additional
`rate_unit` metadata preserves this distinction, without claiming the dashboard
renders a custom physical-unit label.

## mmpOS coverage

The mmpOS adapter installs into a renamed `/opt/mmp/miners/custom-<hash>`
directory and invokes `mmp-launch.sh` from outside that directory. It verifies
coin/protocol/argument translation, the Xelis fallback, explicit overrides,
library paths and exit status. A separate argument recorder is test-only;
the live case always executes the untouched packaged miner.

Production `/mmpos` serialization and the executed `mmp-stats.sh` callback
must agree on CPU identity (`cpu`), full/short PCI addresses, aligned rate and
per-device share arrays, filters, fractional/zero/large rates and aggregate
accepted/invalid/rejected ordering. Negative controls catch unit changes,
swapped devices and mismatched share arrays. Errors produce no stdout stats.
`units=hs` remains the platform scale; Pearl's numbers remain MAC/s.

The live mmpOS case uses the same bounded local CPU/pool test as Hive and is
a separate required CI release dependency. It requires a freshly built binary
containing the mmpOS schema corrections; published v0.9.0 is not sufficient.
Use `run.py --platform mmpos --scenario contracts` inside the isolated test
environment to test current serialization and wrappers without claiming that
an older packaged binary implements the corrected API.

## Source provenance

Reference contract inspected 2026-09-20:

- https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/README.md
- https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/custom
- https://github.com/minershive/hiveos-linux/blob/master/hive/miners/custom/custom-get
- https://github.com/ddobreff/mmpos/blob/main/CUSTOM_MINER.MD
- https://github.com/ddobreff/mmpos/blob/main/custom_miner/mmp-launch.txt

No Hive source is copied or fetched at test time. This is an independently
implemented contract simulator, not Hive's full agent/installer. In particular,
our installer rejects archive traversal and does not implement host upgrades,
ownership changes or service management. Hive's public client carries BSL 1.1:
https://github.com/minershive/hiveos-linux/blob/master/hive/LICENSE.txt
Do not vendor its code under TNN's license; pin revisions and retain upstream
notices if an upstream-backed compatibility mode is added later.

The mmpOS adapter likewise implements the documented contract independently;
it does not vendor or emulate the complete mmpOS agent/dashboard.

The fixture compiles `src/broadcast/broadcastServer.cpp` unchanged, with
test-only definitions of miner globals and the real work-unit/device-filter
headers. It does not duplicate production JSON serialization. Fixture build
uses C++20 and Boost >=1.75; the container uses Ubuntu 24.04's Boost package.

## Local validation receipt (2026-09-20)

18/18 cases passed in a disposable filesystem/network namespace under WSL,
using the published v0.9.0 Linux binary repackaged with the candidate wrappers.
Binary SHA256: `cf1270775a661d730f8a7d2beadc77bab3728c8f7ce1122cb264c519ca786d21`.
The run captured 51 valid live Hive samples, including accepted/rejected shares,
and exited cleanly after 30 mining seconds (about 71 seconds including normal
CPU tuning). ShellCheck 0.10.0 passed. Local receipts are in the ignored
`build/hiveos-local/release-results` directory.

The Docker daemon was unavailable locally, so the Docker image/launcher and
the next newly built release artifact still need their first CI execution.
Do not mistake the WSL validation for a real Hive agent, GPU test, or a rebuild
of the new 0.9.1 source version.

mmpOS follow-up: **22/22 contract cases passed** in the isolated WSL namespace,
using the current production API fixture and candidate packaged wrappers.
Receipts: `build/hiveos-local/mmpos-results`. The Hive suite was rerun and
passed all 18 cases, including live CPU mining. ShellCheck, Python syntax,
CI YAML/release-gate checks and `git diff --check` passed.

The mmpOS **live corrected-API case has not been run locally**: the available
published Linux binary predates the schema fixes. It is implemented and
mandatory in CI against the newly built candidate. No GPU, real mmpOS agent,
or public pool validation is claimed.
