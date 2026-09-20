"""mmpOS contract adapter; run only via the shared isolated harness."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from harness import http_server, require, stop_group
import tnn


def validate(sample, rates=None, buses=None):
    require(sample['units'] == 'hs', 'wrong scale')
    require(len(sample['busid']) == len(sample['hash']), 'device/rate alignment')
    require(all(isinstance(n, (float, int)) and math.isfinite(n) and n >= 0
                for n in sample['hash']), 'invalid rate')
    require(sample['miner_name'] == 'tnn-miner' and sample['miner_version'], 'identity missing')
    require(len(sample['air']) == 3, 'aggregate share ordering')
    for key in ('accepted', 'rejected', 'invalid'):
        require(len(sample['shares'][key]) == len(sample['hash']), 'share alignment')
    if rates is not None:
        require(sample['hash'] == rates, 'rate changed/scaled')
    if buses is not None:
        require(sample['busid'] == buses, 'PCI/CPU identity')


def run(args, results):
    # The real agent renames the archive root using its download URL hash.
    url = 'http://127.0.0.1/' + args.archive.name
    installed = tnn.ROOT / ('custom-' + hashlib.sha256(url.encode()).hexdigest()[:5])
    tnn.INSTALLED.rename(installed)
    launcher = installed / 'mmp-launch.sh'
    stats_script = installed / 'mmp-stats.sh'

    def metadata():
        proc = subprocess.run(['bash', '-c',
                               'source "$1"; printf "%s\\n%s\\n" "$EXTERNAL_NAME" "$EXTERNAL_VERSION"',
                               'mmpos', str(installed / 'mmp-external.conf')],
                              capture_output=True, text=True, timeout=5)
        require(proc.returncode == 0, proc.stderr)
        lines = proc.stdout.splitlines()
        require(len(lines) == 2 and lines[0] == 'tnn-miner' and lines[1], 'invalid package metadata')
    results.test('mmpos-package-metadata', metadata)

    def stats():
        return subprocess.run(['bash', str(stats_script), '99', '/tmp/miner.log'],
                              cwd='/tmp', capture_output=True, text=True, timeout=5)

    def fixture(name, state, rates, buses, devices):
        path = results.directory / (name + '.state.json')
        path.write_text(json.dumps(state))
        proc = subprocess.run([str(args.fixture), str(path), '/mmpos'],
                              capture_output=True, text=True, timeout=10)
        require(proc.returncode == 0, proc.stderr)
        api = json.loads(proc.stdout)
        validate(api, rates, buses)
        require(api['air'] == [7, 0, 2], 'aggregate share order')
        require(api['shares']['accepted'] == [3 if d == 'cpu' else 10 + d for d in devices], 'accepted device mapping')
        require(api['shares']['rejected'] == [1 if d == 'cpu' else 20 + d for d in devices], 'rejected device mapping')
        require(api['shares']['invalid'] == [0] * len(devices), 'invalid device mapping')
        with http_server(8989, api, route='/mmpos'):
            wrapped = stats()
        require(wrapped.returncode == 0, wrapped.stderr)
        require(json.loads(wrapped.stdout) == api, 'stats script changed API data')
        (results.directory / (name + '.mmpos.json')).write_text(json.dumps(api, indent=2))

    gpus = [dict(pci='0000:0a:00.0', rates=[10000, 30000]),
            dict(pci='2a:00.0', rates=[50000, 70000])]
    cases = [
        ('cpu', tnn.state(), [2000], ['cpu'], ['cpu']),
        ('fractional', dict(tnn.state(), cpu_rates=[0, 1]), [0.5], ['cpu'], ['cpu']),
        ('gpu', tnn.state(False, rates=gpus), [20000, 60000], [10, 42], [0, 1]),
        ('hybrid', tnn.state(rates=gpus), [2000, 20000, 60000], ['cpu', 10, 42], ['cpu', 0, 1]),
        ('filtered', tnn.state(False, rates=gpus, include=[1]), [60000], [42], [1]),
        ('excluded', tnn.state(False, rates=gpus, exclude=[0]), [60000], [42], [1]),
        ('reordered', tnn.state(False, rates=gpus[::-1]), [60000, 20000], [42, 10], [0, 1]),
        ('missing-pci', tnn.state(False, rates=[dict(pci='', rates=[4]), dict(pci='invalid', rates=[5])]), [4, 5], [0, 1], [0, 1]),
        ('zero', dict(tnn.state(), cpu_rates=[]), [0], ['cpu'], ['cpu']),
        ('pearl', tnn.state(False, True, [dict(pci='0a:00.0', rates=[52000000000000] * 60)]), [52000000000000], [10], [0]),
        ('overflow', tnn.state(False, True, [dict(pci='0a:00.0', rates=[5000000000000000000] * 3)]), [5000000000000000000], [10], [0]),
    ]
    for case in cases:
        results.test('mmpos-reporting-' + case[0], lambda case=case: fixture(*case))

    def disabled():
        path = results.directory / 'disabled.state.json'
        path.write_text(json.dumps(dict(tnn.state(), mmpos_enabled=False)))
        proc = subprocess.run([str(args.fixture), str(path), '/mmpos'], capture_output=True, timeout=10)
        require(proc.returncode == 3, 'endpoint enabled without --mmpos')
    results.test('mmpos-endpoint-opt-in', disabled)

    def failed_api(payload=None, route='/mmpos'):
        if payload is None:
            proc = stats()
        else:
            with http_server(8989, payload, route=route):
                proc = stats()
        require(proc.returncode != 0 and not proc.stdout and proc.stderr, 'API error must be stderr/nonzero, no JSON')
    results.test('mmpos-api-unavailable', failed_api)
    results.test('mmpos-api-malformed', lambda: failed_api(b'not-json'))
    results.test('mmpos-api-wrong-schema', lambda: failed_api(dict(hash=[])))
    results.test('mmpos-api-http-error', lambda: failed_api({}, '/different'))

    def timed_out_api():
        start = time.monotonic()
        with http_server(8989, {}, route='/mmpos', delay=4):
            proc = stats()
        require(proc.returncode != 0 and not proc.stdout, 'timeout produced statistics')
        require(time.monotonic() - start < 4, 'API request did not respect its deadline')
    results.test('mmpos-api-timeout', timed_out_api)

    def launch_arguments():
        # Argument recorder exists ONLY in a separate test directory. Never
        # overwrite the installed release executable used by the live case.
        scratch = results.directory / 'argument-recorder'
        scratch.mkdir(exist_ok=True)
        shutil.copyfile(launcher, scratch / 'mmp-launch.sh')
        (scratch / 'libs').mkdir(exist_ok=True)
        recorder = scratch / 'tnn-miner'
        recorder.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(args=sys.argv[1:],libs=os.environ.get("LD_LIBRARY_PATH"))))\n')
        recorder.chmod(0o755)
        def invoke(arguments, valid=True):
            proc = subprocess.run(['bash', str(scratch / 'mmp-launch.sh'), *arguments],
                                  cwd='/tmp', capture_output=True, text=True, timeout=5)
            require((proc.returncode == 0) == valid, proc.stderr)
            return json.loads(proc.stdout) if valid else proc
        base = invoke([])
        require(base['args'] == ['--XEL', '--password', 'x', '--mmpos'], 'Xelis default changed')
        require(base['libs'].split(':')[0] == str(scratch / 'libs'), 'bundled libs missing')
        sample = invoke(['--coin', 'XEL', 'tls', '--pool', 'localhost:3333', '--user', 'wallet.worker',
                         '--password', 'a b', '--api-port', '8989', '--threads', '1', '--prl', '--no-cpu'])['args']
        require('--XEL' not in sample and sample.count('--prl') == 1, 'default conflicts with explicit flag')
        require(sample.count('--threads') == 1 and sample[sample.index('--threads') + 1] == '1', 'duplicate threads')
        require(sample[sample.index('--password') + 1] == 'a b', 'argument split')
        require('stratum+ssl://localhost:3333' in sample and '--api-port' not in sample, 'platform argument translation')
        for coin, flag in [('PRL', '--prl'), ('RVN', '--rvn'), ('XMR', '--xmr'), ('DERO', '--dero'), ('SPR', '--spr'), ('unknown', '--XEL')]:
            require(invoke(['--coin', coin])['args'][0] == flag, 'coin mapping: ' + coin)
        require(invoke(['--worker-name', 'tls'])['args'][-2:] == ['--worker-name', 'tls'], 'pass-through value consumed')
        require(invoke(['--algo', '--prl'])['args'][0] == '--prl', 'legacy algo option')
        require('stratum+tcp://localhost:1234' in invoke(['tcp', '--pool', 'localhost:1234'])['args'], 'TCP mapping')
        require('stratum+ssl://localhost:1234' in invoke(['--pool', 'stratum+ssl://localhost:1234'])['args'], 'explicit scheme')
        for option in ('--pool', '--user', '--password', '--coin', '--algo', '--api-port'):
            invoke([option], False)
        proc = subprocess.run(['bash', str(scratch / 'mmp-launch.sh'), '--api-port', '9999'],
                              capture_output=True, text=True, timeout=5)
        require(proc.returncode == 0 and '8989' in proc.stderr, 'API port mismatch not explained')
    results.test('mmpos-launch-arguments', launch_arguments)

    def negative_controls():
        good = json.loads((results.directory / 'gpu.mmpos.json').read_text())
        for key, value in [('hash', [20000000, 60000000]), ('busid', [42, 10])]:
            broken = dict(good, **{key: value})
            try:
                validate(broken, [20000, 60000], [10, 42])
            except AssertionError:
                continue
            raise AssertionError('negative control passed: ' + key)
        broken = dict(good, shares={'accepted': [1], 'rejected': [0], 'invalid': [0]})
        failed_api(broken)
    results.test('mmpos-negative-controls', negative_controls)

    def launch_failure():
        with (results.directory / 'invalid-launch.log').open('w') as log:
            proc = subprocess.Popen(['bash', str(launcher), '--this-is-an-invalid-ci-option'],
                                    cwd='/tmp', stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                require(proc.wait(timeout=15) != 0, 'launcher hid miner failure')
            finally:
                stop_group(proc)
    results.test('mmpos-launch-failure', launch_failure)

    def live():
        samples = []
        with tnn.pool() as (port, packets, errors):
            command = ['bash', str(launcher), '--coin', 'XEL', 'tcp', '--pool', f'127.0.0.1:{port}',
                       '--user', 'ci-only', '--password', 'x', '--api-port', '8989', '--xel-v3',
                       '--no-gpu', '--threads', '1', '--no-msr', '--no-lock', '--xelis-simd', 'none',
                       '--ignore-wallet', '--worker-name', 'mmpos-ci', '--report-interval', '1', '--mine-time', '30']
            with (results.directory / 'miner.log').open('w') as log:
                proc = subprocess.Popen(command, cwd='/tmp', stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 90
                try:
                    while proc.poll() is None and time.monotonic() < deadline:
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:8989/mmpos', timeout=2) as response:
                                api = json.load(response)
                        except OSError:
                            time.sleep(0.25)
                            continue
                        validate(api, buses=['cpu'])
                        wrapped = stats()
                        require(wrapped.returncode == 0, wrapped.stderr)
                        sample = json.loads(wrapped.stdout)
                        validate(sample, buses=['cpu'])
                        require(api['miner_version'] == results.metadata['expected_version'], 'live release version')
                        samples.append(dict(api=api, wrapped=sample))
                        time.sleep(0.25)
                    require(proc.poll() is not None, 'miner exceeded deadline')
                    require(proc.returncode == 0, 'miner did not exit cleanly')
                finally:
                    stop_group(proc)
                    (results.directory / 'live-mmpos.json').write_text(json.dumps(samples, indent=2))
                    (results.directory / 'stratum.json').write_text(json.dumps(packets, indent=2))
            require(not errors, str(errors))
            require(any(p.get('method') == 'mining.authorize' for p in packets), 'no authorization')
            for source in ('api', 'wrapped'):
                require(any(s[source]['hash'][0] > 0 for s in samples), 'no live rate')
                require(any(s[source]['air'][0] > 0 and s[source]['air'][2] > 0 for s in samples), 'missing share responses')
                require(any(s[source]['shares']['accepted'][0] > 0 and
                            s[source]['shares']['rejected'][0] > 0 for s in samples),
                        'missing live per-device share responses')
    if args.scenario == 'all':
        results.test('mmpos-packaged-miner-local-cpu-mining', live)
