#!/usr/bin/env python3
"""Check the installed OpenCode transport against a local mock, without secrets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import select
import socket
import subprocess
import tempfile
import threading
import time
import uuid


def check_transport(image: str, mode: str, scratch: Path) -> dict:
    requests: list[float] = []
    request_hashes: list[str] = []
    disconnects: list[float] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(time.monotonic())
            request_hashes.append(hashlib.sha256(body).hexdigest())
            try:
                if mode == "headers":
                    time.sleep(2.5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {"id":"canary", "object":"chat.completion.chunk",
                         "created":1, "model":"fixture",
                         "choices":[{"index":0,"delta":{"role":"assistant",
                             "content":"CANARY_OK"},"finish_reason":None}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                if mode in ("chunks", "stream_control", "total"):
                    sent_at = time.monotonic()
                    while time.monotonic() - sent_at < 2.5:
                        if mode == "total":
                            chunk["choices"][0]["delta"] = {"content":" "}
                            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                            self.wfile.flush()
                            time.sleep(0.15)
                        readable, _, _ = select.select([self.connection], [], [], 0.05)
                        if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                            disconnects.append(time.monotonic() - sent_at)
                            return
                chunk["choices"] = [{"index":0,"delta":{},"finish_reason":"stop"}]
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n"
                                  "data: [DONE]\n\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    directory = Path(tempfile.mkdtemp(prefix="opencode-transport-", dir=scratch))
    config_dir = directory / ".config/opencode"
    config_dir.mkdir(parents=True)
    config = {
        "$schema":"https://opencode.ai/config.json",
        "model":"fixture/fixture", "small_model":"fixture/fixture",
        "enabled_providers":["fixture"],
        "provider":{"fixture":{"npm":"@ai-sdk/openai-compatible",
            "options":{"baseURL":f"http://127.0.0.1:{server.server_port}/v1",
                "apiKey":"dummy-canary", "timeout":750 if mode == "total" else 6000,
                "headerTimeout":750,
                "chunkTimeout":5000 if mode in ("stream_control","total") else 750},
            "models":{"fixture":{"name":"fixture",
                "limit":{"context":32768,"output":256}}}}},
    }
    (config_dir / "opencode.json").write_text(json.dumps(config))
    name = "agsci-transport-" + uuid.uuid4().hex[:12]
    command = ["docker","run","--rm","--name",name,
               "--network","host","--user","1000:1000",
               "-e","HOME=/canary","-v",f"{directory}:/canary","-w","/canary",
               "--entrypoint","opencode",image,"run","--format","json",
               "--model","fixture/fixture","Reply only CANARY_OK. Do not use tools."]
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    observed_timeout = False
    timeout_log_message = False
    stopped_after_assertion = False
    try:
        if mode not in ("success", "stream_control"):
            deadline = start + 20
            while time.monotonic() < deadline and process.poll() is None:
                logs = list(directory.glob(".local/share/opencode/log/*.log"))
                tail = "\n".join(p.read_text(errors="replace")[-24000:] for p in logs)
                timeout_log_message = any(word in tail.lower() for word in
                                          ("timed out", "timeout error", "timeouterror"))
                # Some OpenCode builds retry an interrupted response without
                # printing a stream error. The local server also witnesses
                # cancellation before its deliberately delayed next chunk.
                early_disconnect = any(0.35 <= delay < 1.75 for delay in disconnects)
                repeated_request = max(Counter(request_hashes).values(), default=0) >= 2
                # The slow-stream control completes the same response under a
                # longer chunk deadline. Reissuing the identical request with
                # only a short deadline is a second, observable abort witness.
                retry_witness = mode in ("chunks", "total") and repeated_request
                observed_timeout = timeout_log_message or early_disconnect or retry_witness
                if observed_timeout:
                    # The CLI retries transport errors. The canary proves the
                    # first timer fired; do not wait for all retry backoffs.
                    stopped_after_assertion = True
                    subprocess.run(["docker","stop","--time","1",name],
                                   capture_output=True, timeout=10)
                    break
                time.sleep(0.2)
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker","stop","--time","1",name],
                       capture_output=True, timeout=10)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            subprocess.run(["docker","stop","--time","1",name],
                           capture_output=True, timeout=10)
        server.shutdown()
        server.server_close()
    elapsed = time.monotonic() - start
    output = stdout + stderr
    timeout_message = timeout_log_message or any(
        word in output.lower() for word in ("timed out", "timeout", "abort"))
    timeout_signal = observed_timeout or timeout_message
    if mode in ("success", "stream_control"):
        passed = bool(requests) and process.returncode == 0 and "CANARY_OK" in output
    else:
        passed = bool(requests) and timeout_signal
    receipt = {"mode":mode,"passed":passed,"elapsed_seconds":elapsed,
               "request_count":len(requests),"return_code":process.returncode,
               "timeout_message_observed":timeout_message,
               "timeout_signal_observed":timeout_signal,
               "server_disconnect_seconds":disconnects,
               "identical_request_max_count":max(Counter(request_hashes).values(), default=0),
               "mock_body_delay_seconds":2.5 if mode in ("chunks","stream_control","total") else 0,
               "stopped_after_expected_timeout":stopped_after_assertion,
               "network":"local mock only","credentials":"dummy only",
               "gpu":"none","medical_data_mounted":False}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # The fixture has no real credentials; preserve full diagnostics on failure.
    if not passed:
        (directory / "failure.log").write_text(output)
        receipt["failure_log"] = str(directory / "failure.log")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = [check_transport(args.image, mode, args.scratch)
                for mode in ("success", "stream_control", "headers", "chunks", "total")]
    report = {"passed":all(x["passed"] for x in receipts),"image_id":args.image,
              "checks":receipts}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
