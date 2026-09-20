"""TNN adapter: fixture state, Hive callbacks, assertions, local Xelis protocol."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import socketserver
import subprocess
import threading
from harness import require

MINER = 'tnn-miner'
ROOT = Path('/hive/miners/custom')
INSTALLED = ROOT / MINER


def stats(package=INSTALLED):
    # Like Hive's agent, source the callback and read its variables rather than
    # interpreting informational stdout as the report. Seed stale state on purpose.
    command = '''
get_miner_uptime() { echo 120; }
cpu-temp() { echo 55; }
log_name=/var/log/miner/custom/custom.log
log_head_name=/tmp/log-head
khs=999999
stats='{"stale":true}'
source "$1/h-stats.sh" >&2
status=$?
printf '{"callback_status":%d,"khs":%s,"stats":%s}\\n' "$status" "$khs" "$stats"
'''
    result = subprocess.run(['bash', '-c', command, 'hive-agent', str(package)],
                            capture_output=True, text=True, timeout=8)
    require(result.returncode == 0, result.stderr)
    try:
        return json.loads(result.stdout)
    except ValueError as error:
        raise AssertionError('invalid wrapper JSON: ' + result.stdout + '\n' + result.stderr) from error


def configure(extra, package=INSTALLED):
    import os
    env = dict(os.environ, CUSTOM_USER_CONFIG=extra, CUSTOM_MINER=MINER,
               CUSTOM_ALGO='xelishashv3', CUSTOM_URL='127.0.0.1', CUSTOM_TEMPLATE='ci-only')
    result = subprocess.run(['bash', '-c', 'source ./h-manifest.conf; source ./h-config.sh'],
                            cwd=package, env=env, capture_output=True, text=True, timeout=5)
    require(result.returncode == 0, result.stderr)
    require((package / 'config.conf').read_text().strip() == extra, 'flight-sheet args changed')
    return result.stdout + result.stderr


def state(cpu=True, pearl=False, rates=None, include=None, exclude=None):
    return dict(cpu=cpu, pearl=pearl, cpu_rates=[1000, 3000] if cpu else [],
                gpus=rates or [], include=include or [], exclude=exclude or [])


def validate(sample, hs, buses, temps, fans, unit='H/s', accepted=7, rejected=2):
    require(sample['callback_status'] == 0, 'stats callback failed')
    value = sample['stats']
    require(value['hs'] == hs, 'per-device rates differ')
    require(value['bus_numbers'] == buses, 'device identities differ')
    require(value['temp'] == temps and value['fan'] == fans, 'telemetry associated with wrong device')
    require(value['ar'] == [accepted, rejected], 'share counters differ')
    require(value['hs_units'] == 'hs' and value['rate_unit'] == unit, 'rate units differ')
    require(math.isclose(float(sample['khs']), sum(hs) / 1000, abs_tol=0.011), 'aggregate scaling differs')


@contextmanager
def pool():
    """Protocol fixture, NOT a network consensus/proof validator."""
    records, errors = [], []
    lock = threading.Lock()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(45)
            try:
                while True:
                    line = self.rfile.readline(65536)
                    if not line:
                        break
                    request = json.loads(line)
                    method = request.get('method')
                    with lock:
                        records.append(request)
                    def send(value):
                        self.wfile.write(json.dumps(value).encode() + b'\n')
                        self.wfile.flush()
                    reply = dict(id=request.get('id'), result=True, error=None)
                    if method == 'mining.subscribe':
                        reply['result'] = [[], '00' * 8, 8, '11' * 32]
                        send(reply)
                    elif method == 'mining.authorize':
                        send(reply)
                        send(dict(method='mining.set_difficulty', params=[1.0]))
                        send(dict(method='mining.notify', params=['ci-job', '01', '22' * 32, 0, True]))
                    elif method == 'mining.submit':
                        require(isinstance(request.get('params'), list), 'malformed submission')
                        with lock:
                            count = sum(r.get('method') == method for r in records)
                        if count == 2:
                            reply.update(result=False, error={'message': 'CI scripted rejection'})
                        send(reply)
                    else:
                        send(reply)
            except (ConnectionError, TimeoutError):
                pass
            except Exception as error:
                errors.append(str(error))

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = Server(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        yield server.server_address[1], records, errors
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
