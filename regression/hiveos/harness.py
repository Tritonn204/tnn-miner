"""Miner-independent package, HTTP, process and result helpers."""
from contextlib import contextmanager
import hashlib
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_binary_digest(path, expected):
    require(sha256(path) == expected, 'wrapper contains wrong binary')


def package_members(archive, miner):
    """Reject traversal, special files and escaping symlinks before extraction."""
    import posixpath
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = {}
        links = set()
        for member in members:
            name = posixpath.normpath(member.name)
            require(name == miner or name.startswith(miner + '/'), 'archive escapes miner directory')
            require(name not in names, 'duplicate archive entry: ' + name)
            require(member.isfile() or member.isdir() or member.issym(), 'unsupported archive member')
            if member.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
                require(target.startswith(miner + '/'), 'symlink escapes miner directory')
                links.add(name)
            names[name] = member
        for name in names:
            require(not any(name.startswith(link + '/') for link in links), 'entry beneath symlink')
        return names


def install(archive, destination, miner, callbacks=('h-config.sh', 'h-run.sh', 'h-stats.sh', 'h-manifest.conf')):
    names = package_members(archive, miner)
    for filename in (*callbacks, miner):
        key = miner + '/' + filename
        require(key in names and names[key].isfile(), 'missing package file: ' + key)
        if filename.endswith('.sh') or filename == miner:
            require(names[key].mode & 0o111, 'not executable: ' + key)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter='data')
    return Path(destination) / miner


@contextmanager
def http_server(port=0, payload=None, directory=None, route='/stats', delay=0):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if delay:
                time.sleep(delay)
            if directory is not None:
                from urllib.parse import unquote, urlparse
                name = Path(unquote(urlparse(self.path).path)).name
                target = Path(directory) / name
                if not target.is_file():
                    self.send_error(404)
                    return
                body = target.read_bytes()
            else:
                if self.path != route:
                    self.send_error(404)
                    return
                body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when testing the client's request deadline.

        def log_message(self, *args):
            pass

    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server(('127.0.0.1', port), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05})
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def stop_group(process):
    """Always reap the wrapper, tee, and miner, including failed test paths."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    process.wait(timeout=2)


class Results:
    def __init__(self, directory, name='hiveos-integration'):
        self.name = name
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.metadata = {}

    def test(self, name, callback):
        start = time.monotonic()
        error = None
        try:
            callback()
        except Exception:
            error = traceback.format_exc()
        self.rows.append(dict(name=name, seconds=time.monotonic() - start, error=error))
        print(('FAIL ' if error else 'PASS ') + name, flush=True)
        if error:
            print(error, flush=True)
        self.save()
        return error is None

    def save(self):
        failures = sum(bool(row['error']) for row in self.rows)
        root = ET.Element('testsuite', name=self.name, tests=str(len(self.rows)), failures=str(failures))
        for row in self.rows:
            case = ET.SubElement(root, 'testcase', name=row['name'], time=str(row['seconds']))
            if row['error']:
                ET.SubElement(case, 'failure').text = row['error']
        ET.ElementTree(root).write(self.directory / 'junit.xml', encoding='utf-8', xml_declaration=True)
        (self.directory / 'report.json').write_text(json.dumps(
            dict(metadata=self.metadata, tests=self.rows, failures=failures), indent=2))

    @property
    def passed(self):
        return bool(self.rows) and not any(row['error'] for row in self.rows)
