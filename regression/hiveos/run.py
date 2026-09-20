"""Run ONLY inside a disposable Linux container/network namespace.

The installation paths deliberately match Hive; never run on a real Hive rig.
"""
import argparse
import io
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request

from harness import Results, http_server, install, require, sha256, stop_group, verify_binary_digest
import tnn


def read_api():
    with urllib.request.urlopen('http://127.0.0.1:8989/stats', timeout=2) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--linux-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/out'))
    parser.add_argument('--fixture', type=Path, default=Path('/fixture/stats_fixture'))
    parser.add_argument('--scenario', choices=('all', 'contracts'), default='all')
    parser.add_argument('--platform', choices=('hiveos', 'mmpos'), default='hiveos')
    args = parser.parse_args()
    require(os.environ.get('HIVEOS_TEST_ISOLATED') == '1', 'use ci.sh; isolated container required')
    results = Results(args.output, args.platform + '-integration')
    if args.platform == 'mmpos':
        # Share artifact identity checks; use a platform-specific installation.
        tnn.ROOT = Path('/opt/mmp/miners')
        tnn.INSTALLED = tnn.ROOT / tnn.MINER
    dependencies = args.fixture.parent / 'dependencies.tsv'
    if dependencies.exists():
        shutil.copyfile(dependencies, results.directory / 'dependencies.tsv')
    results.metadata.update(archive_sha256=sha256(args.archive), scenario=args.scenario, platform=args.platform,
                            real_gpu_execution=False, proof_validation=False)
    tnn.ROOT.mkdir(parents=True, exist_ok=True)
    Path('/run/hive').mkdir(parents=True, exist_ok=True)
    Path('/var/log/miner/custom').mkdir(parents=True, exist_ok=True)

    def packaged():
        require(args.archive.name.startswith('tnn-miner-') and
                args.archive.name.endswith('.hiveos_mmpos.amd64.tar.gz'), 'not the Hive release archive')
        # Exercise an installation URL rather than silently substituting source files.
        with http_server(directory=args.archive.parent) as port:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/{args.archive.name}') as response:
                downloaded = Path('/tmp') / args.archive.name
                downloaded.write_bytes(response.read())
        require(sha256(downloaded) == sha256(args.archive), 'download changed archive')
        callbacks = ('mmp-launch.sh', 'mmp-stats.sh', 'mmp-external.conf') if args.platform == 'mmpos' else (
            'h-config.sh', 'h-run.sh', 'h-stats.sh', 'h-manifest.conf')
        package = install(downloaded, tnn.ROOT, tnn.MINER, callbacks)
        with tarfile.open(args.linux_archive) as tar:
            candidates = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == 'tnn-miner']
            require(len(candidates) == 1, 'ambiguous Linux build binary')
            import hashlib
            expected = hashlib.sha256(tar.extractfile(candidates[0]).read()).hexdigest()
            verify_binary_digest(package / tnn.MINER, expected)
            # Bundled libraries/tunes must also survive packaging byte-for-byte.
            for member in tar.getmembers():
                relative = member.name.removeprefix('./')
                if member.isfile() and relative.startswith(('libs/', 'tunes/')):
                    target = package / relative
                    require(target.is_file(), 'missing bundled file: ' + relative)
                    require(hashlib.sha256(target.read_bytes()).digest() ==
                            hashlib.sha256(tar.extractfile(member).read()).digest(), 'bundle changed: ' + relative)
        results.metadata['binary_sha256'] = expected
        results.metadata['fixture_sha256'] = sha256(args.fixture)
        version = re.fullmatch(r'Tnn-miner-amd64-(.+)\.tar\.gz', args.linux_archive.name)
        if args.scenario == 'all':
            require(version is not None, 'cannot determine expected release version')
            results.metadata['expected_version'] = re.sub(r'^[a-zA-Z]', '', version[1])
        require(not any(p.name in ('stats_fixture', 'harness.py', 'run.py') for p in package.rglob('*')),
                'test code shipped in archive')

    if not results.test('release-package-install-and-identity', packaged):
        return 1

    if args.platform == 'mmpos':
        import mmpos
        mmpos.run(args, results)
        results.save()
        return 0 if results.passed else 1

    results.test('flight-sheet-config', lambda: tnn.configure('--xel-v3 --no-gpu --threads 1 --worker-name ci-worker'))

    def fixture_case(name, state, hs, buses, temps, fans, telemetry=True):
        path = results.directory / (name + '.state.json')
        path.write_text(json.dumps(state))
        proc = subprocess.run([str(args.fixture), str(path)], capture_output=True, text=True, timeout=10)
        require(proc.returncode == 0, proc.stderr)
        api = json.loads(proc.stdout)
        require(math.isclose(api['hashrate'], sum(hs)), 'production API aggregate differs')
        (results.directory / (name + '.api.json')).write_text(json.dumps(api, indent=2))
        hive = Path('/run/hive/gpu-stats.json')
        if telemetry:
            # Intentionally reverse the system inventory relative to API order.
            hive.write_text(json.dumps(dict(busids=['0000:2a:00.0', '0000:0a:00.0'],
                                            temp=[72, 61], fan=[80, 40])))
        elif hive.exists():
            hive.unlink()
        with http_server(8989, api):
            sample = tnn.stats()
        (results.directory / (name + '.hive.json')).write_text(json.dumps(sample, indent=2))
        tnn.validate(sample, hs, buses, temps, fans, 'MAC/s' if state['pearl'] else 'H/s')
        require(sample['stats']['uptime'] >= 120, 'uptime missing')
        require(sample['stats']['ver'] == 'fixture-v1', 'version missing')
        return sample

    gpus = [dict(pci='0000:0a:00.0', rates=[10000, 30000]),
            dict(pci='0000:2a:00.0', rates=[50000, 70000])]
    cases = [
        ('cpu', tnn.state(), [2000], [None], [55], [None], True),
        ('cpu-sub-kilohash', dict(tnn.state(), cpu_rates=[400, 600]), [500], [None], [55], [None], True),
        ('gpu', tnn.state(False, rates=gpus), [20000, 60000], [10, 42], [61, 72], [40, 80], True),
        ('hybrid', tnn.state(rates=gpus), [2000, 20000, 60000], [None, 10, 42], [55, 61, 72], [None, 40, 80], True),
        ('filtered', tnn.state(False, rates=gpus, include=[1]), [60000], [42], [72], [80], True),
        ('excluded', tnn.state(False, rates=gpus, exclude=[0]), [60000], [42], [72], [80], True),
        ('reordered', tnn.state(False, rates=list(reversed(gpus))), [60000, 20000], [42, 10], [72, 61], [80, 40], True),
        ('missing-telemetry', tnn.state(False, rates=gpus), [20000, 60000], [10, 42], [0, 0], [0, 0], False),
        ('zero-startup', dict(tnn.state(), cpu_rates=[]), [0], [None], [55], [None], True),
        ('pearl', tnn.state(False, True, [dict(pci='0000:0a:00.0', rates=[52000000000000])]),
         [52000000000000], [10], [61], [40], True),
    ]
    for case in cases:
        results.test('reporting-' + case[0], lambda case=case: fixture_case(*case))

    def failed_api(payload):
        if payload is None:
            sample = tnn.stats()
        else:
            with http_server(8989, payload):
                sample = tnn.stats()
        require(sample == dict(callback_status=1, khs=0, stats=None), 'failed API retained stale statistics')
    results.test('api-unavailable', lambda: failed_api(None))
    results.test('api-malformed', lambda: failed_api(b'not-json'))
    results.test('api-wrong-schema', lambda: failed_api(dict(hashrate='broken')))

    def negative_controls():
        # Mutation controls prove the validators fail, not merely the happy path.
        def rejects(action):
            try:
                action()
            except (AssertionError, ValueError):
                return
            raise AssertionError('negative control unexpectedly passed')
        broken = results.directory / 'broken.tar.gz'
        with tarfile.open(broken, 'w:gz') as tar:
            member = tarfile.TarInfo('tnn-miner/h-manifest.conf')
            member.size = 1
            tar.addfile(member, io.BytesIO(b'x'))
        rejects(lambda: install(broken, results.directory / 'negative', tnn.MINER))
        sample = json.loads((results.directory / 'gpu.hive.json').read_text())
        original = sample['khs']
        sample['khs'] *= 1000
        rejects(lambda: tnn.validate(sample, [20000, 60000], [10, 42], [61, 72], [40, 80]))
        sample['khs'] = original
        sample['stats']['temp'].reverse()
        rejects(lambda: tnn.validate(sample, [20000, 60000], [10, 42], [61, 72], [40, 80]))
        rejects(lambda: verify_binary_digest(tnn.INSTALLED / tnn.MINER, '0' * 64))
    results.test('negative-controls', negative_controls)

    def launch_failure():
        tnn.configure('--this-is-an-invalid-ci-option')
        with (results.directory / 'invalid-launch.log').open('w') as log:
            process = subprocess.Popen(['bash', './h-run.sh'], cwd=tnn.INSTALLED,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                require(process.wait(timeout=15) != 0, 'wrapper hid miner failure')
            finally:
                stop_group(process)
    results.test('launch-failure-propagation', launch_failure)

    def live():
        snapshots = []
        wrapped_samples = []
        with tnn.pool() as (port, packets, errors):
            extra = (f'--xel-v3 --no-gpu --threads 1 --no-msr --no-lock --xelis-simd none '
                     f'--daemon-address stratum+tcp://127.0.0.1 --port {port} '
                     '--wallet ci-only --ignore-wallet --worker-name hive-ci --password x '
                     '--report-interval 1 --mine-time 30')
            (results.directory / 'flight-sheet.log').write_text(tnn.configure(extra))
            with (results.directory / 'miner.log').open('w') as log:
                process = subprocess.Popen(['bash', './h-run.sh'], cwd=tnn.INSTALLED,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 90
                try:
                    while process.poll() is None and time.monotonic() < deadline:
                        try:
                            api = read_api()
                        except OSError:
                            time.sleep(0.25)  # API may not be listening during startup.
                            continue
                        snapshots.append(api)
                        if api.get('hashrate', 0) > 0:
                            wrapped = tnn.stats()
                            require(wrapped['callback_status'] == 0, 'live stats callback failed')
                            # Polls can straddle updates; compare shape/units here,
                            # exact numerical transformations in deterministic tests.
                            require(wrapped['stats']['hs_units'] == 'hs', 'live scaling units')
                            require(wrapped['stats']['rate_unit'] == 'H/s', 'live physical unit')
                            require(wrapped['stats']['bus_numbers'] == [None], 'live CPU identity')
                            wrapped_samples.append(wrapped)
                        time.sleep(0.25)
                    require(process.poll() is not None, 'miner exceeded 90-second deadline')
                    require(process.returncode == 0, 'miner/wrapper did not exit cleanly')
                finally:
                    stop_group(process)
                    (results.directory / 'live-api.json').write_text(json.dumps(snapshots, indent=2))
                    (results.directory / 'live-hive.json').write_text(json.dumps(wrapped_samples, indent=2))
                    (results.directory / 'stratum.json').write_text(json.dumps(packets, indent=2))
            require(not errors, str(errors))
            require(any(p.get('method') == 'mining.authorize' for p in packets), 'no local authorization')
            require(snapshots, 'no API snapshots')
            require(wrapped_samples, 'no successfully parsed live wrapper samples')
            require(any(w['stats']['ar'][0] >= 1 and w['stats']['ar'][1] >= 1 for w in wrapped_samples),
                    'share responses not reflected in Hive stats')
            good = [s for s in snapshots if s.get('cpu_hashrate', 0) > 0]
            require(good, 'no nonzero CPU rate')
            for s in good:
                require(math.isfinite(s['hashrate']) and s['hashrate'] == s['cpu_hashrate'], 'live aggregate mismatch')
                require(not s.get('gpus'), 'unexpected GPU mining')
                require(s['algo'] == 'XelisHashV3' and
                        s['version'] == results.metadata['expected_version'], 'live version/algorithm')
            require(max(s['uptime'] for s in snapshots) > min(s['uptime'] for s in snapshots), 'uptime did not advance')
            require(any(s['accepted'] >= 1 and s['rejected'] >= 1 for s in snapshots), 'share responses not reflected in API')
    if args.scenario == 'all':
        results.test('packaged-miner-local-cpu-mining', live)
    results.save()
    return 0 if results.passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
