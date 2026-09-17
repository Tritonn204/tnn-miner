"""Compile a CPU-only client around TNN's actual session and test loopback I/O."""
from pathlib import Path
import json
import re
import socket
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "hip-build/win32/orochi"
OUT = ROOT / "build/pearl-transport-tests"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cached = sorted((ROOT / "_deps").glob("CPM-cpu-Windows-Clang_20_1_8/boost/*/libs"))
    includes = [ROOT / "include", ROOT / "src", Path("C:/openssl/clang64/include")]
    includes.extend(path for path in cached[0].rglob("include") if path.is_dir())
    executable = OUT / "test_transport.exe"
    libraries = [BUILD / "_deps/boost-build/libs" / name / f"libboost_{name}.a"
                 for name in ("json", "context", "chrono", "atomic")]
    command = ["C:/mingw64/bin/clang++.exe", "-std=c++20", "-O2", "-DBOOST_ALL_NO_LIB",
               *[f"-I{path}" for path in includes],
               str(Path(__file__).with_name("test_transport.cpp")),
               str(BUILD / "CMakeFiles/tnn-miner.dir/src/net/pearl/net_pearl_stratum.cpp.obj"),
               *map(str, libraries), "C:/openssl/clang64/lib/libssl.a", "C:/openssl/clang64/lib/libcrypto.a",
               "-lws2_32", "-lcrypt32", "-ladvapi32", "-lwinpthread", "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (OUT / "build.log").write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError(build.stdout + build.stderr)
    logging = subprocess.run([str(executable), "--logging"], capture_output=True, text=True, timeout=5)
    logging.check_returncode()
    clean = re.sub(r"\x1b\[[0-9;]*m", "", logging.stdout)
    assert clean == ("STATUS >> \n[PEARL-STRATUM] GPU 3 share accepted\n"
                     "\n[PEARL-STRATUM] GPU 3 share rejected: bad target\n"
                     "\nDEV | [PEARL-STRATUM] GPU 3 share accepted\n"
                     "\nDEV | [PEARL-STRATUM] GPU 3 share rejected: stale\nQUIETEND")
    assert logging.stderr == "\n[PEARL-STRATUM] test error\n"
    errors = []
    packets = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(12)

        def server():
            try:
                first, _ = listener.accept()
                with first:
                    first.settimeout(10)
                    stream = first.makefile("rb")
                    assert json.loads(stream.readline())["method"] == "mining.authorize"
                    stream.close()
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(10)
                    stream = connection.makefile("rb")
                    auth = json.loads(stream.readline())
                    assert auth["method"] == "mining.authorize" and isinstance(auth["params"], dict)
                    packets.append(auth)
                    notify = {"method": "mining.notify", "params": {"job_id": "transport-job",
                        "header": "00" * 76, "target": "ff" * 32, "height": 1, "cert_version": 2}}
                    payload = (json.dumps(notify) + "\n" + json.dumps({"id": 1, "result": True, "error": None}) + "\n").encode()
                    # Fragmented notify followed by a coalesced notify tail/auth ack.
                    connection.sendall(payload[:17])
                    connection.sendall(payload[17:])
                    submitted = json.loads(stream.readline())
                    assert set(submitted) == {"id", "method", "params"}
                    assert submitted["method"] == "mining.submit" and submitted["id"] > 1
                    assert submitted["params"] == {"job_id": "transport-job", "plain_proof": "AAAA"}
                    packets.append(submitted)
                    connection.sendall((json.dumps({"id": submitted["id"], "result": None,
                        "error": {"code": 21, "msg": "stale - chain tip advanced"}}) + "\n").encode())
                    submitted = json.loads(stream.readline())
                    assert submitted["method"] == "mining.submit"
                    packets.append(submitted)
                    connection.sendall((json.dumps({"id": submitted["id"], "result": True, "error": None}) + "\n").encode())
                    while connection.recv(1024):
                        pass
            except Exception as error:
                errors.append(repr(error))

        worker = threading.Thread(target=server, daemon=True)
        worker.start()
        result = subprocess.run([str(executable), str(listener.getsockname()[1])],
                                capture_output=True, text=True, timeout=15)
        worker.join(timeout=2)
        (OUT / "test.log").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, end="")
        result.check_returncode()
        for marker in ("[PEARL-JOB]", "[PEARL-ACK]", "[PEARL-SUBMIT]", "[PEARL-SUBMIT-BIND]"):
            assert marker not in result.stdout, f"diagnostic leaked into normal logging: {marker}"
        assert "share accepted" in result.stdout and "share rejected" in result.stdout
        if errors or worker.is_alive():
            raise RuntimeError(str(errors) or "loopback server did not stop")
    (OUT / "report.json").write_text(json.dumps({"status": "actual-session-loopback-pass",
        "gpu_launches": 0, "proof_validation": False, "packets": packets}, indent=2))


if __name__ == "__main__":
    main()
