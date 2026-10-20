"""Per-call Python MCP service, owned by the host review lifecycle."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
from pathlib import Path
import re
import secrets
import sys
import threading
from typing import Any

from . import execution_bridge, finite_python
from .agent_tool_profile import PrivateToolProfile, _private, _regular
from .finite_python import FinitePythonBroker, MAX_AUDIT_BYTES, POLICY, execution_policy
from .source_lookup import _json, _parse, _sha

DIRECTORY = "python_execution"


def tools() -> list[dict[str, Any]]:
    return [
        {"name": "get_execution_policy", "description": "Read the fixed image and CPU execution limits.",
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "run_python", "description": "Run your Python source in a fresh isolated CPU container. INPUT_DATA is the optional input JSON object. No network, GPU, host credentials or Docker socket are provided. Actual output and receipts do not establish scientific validity.",
         "inputSchema": {"type": "object", "properties": {"code": {"type": "string", "maxLength": 32768}, "input": {"type": "object"}}, "required": ["code"], "additionalProperties": False}},
    ]


def _tool_result(value: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": _json(value)}],
            "isError": value.get("status") not in {None, "completed"}}


class PreparedExecutionTools:
    def __init__(self, *, artifacts: Path, temporary: Path, profile: PrivateToolProfile,
                 image: str, prompt_sha256: str, cancel_event=None):
        self.root = artifacts / DIRECTORY
        self.profile = profile
        self.stop_event = threading.Event()
        self.closed = False
        self.initialized = False
        self.listed = False
        self.request_count = 0
        self.audit_bytes = 0
        owner = self

        class Cancellation:
            def is_set(self) -> bool:
                return owner.stop_event.is_set() or (cancel_event is not None and cancel_event.is_set())

        self.broker = FinitePythonBroker(self.root, image=image, prompt_sha256=prompt_sha256,
                                         cancel_event=Cancellation())
        self.installed = {"runner.py": Path(finite_python.__file__).resolve(),
                          "bridge.py": Path(execution_bridge.__file__).resolve(),
                          "adapter.py": Path(__file__).resolve()}
        self.module_hashes = {}
        self.audit_path = self.root / "calls.jsonl"
        try:
            for filename, installed in self.installed.items():
                body = _regular(installed, MAX_AUDIT_BYTES)
                _private(self.root / filename, body)
                self.module_hashes[filename] = _sha(body)
            _private(self.audit_path, b"")
            token = secrets.token_hex(32)
            self.token_path = temporary / "python-execution-token"
            _private(self.token_path, token.encode())

            class Handler(BaseHTTPRequestHandler):
                def setup(self):
                    super().setup()
                    self.connection.settimeout(3)

                def log_message(self, *_args):
                    pass

                def reply(self, status, data=None):
                    body = _json(data).encode() if data is not None else b""
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass  # Cancellation can close the native bridge first.

                def authorized(self) -> bool:
                    authorization = self.headers.get("Authorization", "")
                    if not authorization.isascii() or not hmac.compare_digest(authorization, "Bearer " + token):
                        self.reply(401)
                        return False
                    host = f"127.0.0.1:{self.server.server_port}"
                    if self.headers.get("Host") != host or self.headers.get("Origin") not in {None, "http://" + host}:
                        self.reply(403)
                        return False
                    return True

                def do_POST(self):
                    self.connection.settimeout(3)
                    if self.path != "/mcp":
                        return self.reply(404)
                    if not self.authorized():
                        return
                    if self.headers.get("MCP-Protocol-Version", "2025-06-18") not in {"2025-03-26", "2025-06-18"}:
                        return self.reply(400)
                    if owner.stop_event.is_set() or owner.request_count >= 64:
                        return self.reply(503)
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 131072 or self.headers.get("Transfer-Encoding"):
                            raise ValueError("Invalid request size")
                        raw = self.rfile.read(length)
                        if len(raw) != length:
                            raise ValueError("Incomplete request")
                        request = _parse(raw.decode())
                        if request.get("jsonrpc") != "2.0":
                            raise ValueError("Invalid JSON-RPC")
                        if "id" not in request:
                            return self.reply(202)
                        if type(request["id"]) not in {str, int}:
                            raise ValueError("Invalid request identity")
                        owner.request_count += 1
                        result = owner.dispatch(request)
                        event = {"prompt_sha256": prompt_sha256, "request": request,
                                 "request_sha256": _sha(_json(request)), "result": result,
                                 "result_sha256": _sha(_json(result))}
                        line = (_json(event) + "\n").encode()
                        if owner.audit_bytes + len(line) > MAX_AUDIT_BYTES:
                            raise ValueError("Execution audit size exceeded")
                        with owner.audit_path.open("ab") as stream:
                            stream.write(line)
                        owner.audit_bytes += len(line)
                    except (ValueError, OSError, UnicodeError, RecursionError):
                        return self.reply(400)
                    self.reply(200, {"jsonrpc": "2.0", "id": request["id"], "result": result})

                def do_GET(self):
                    if self.authorized():
                        self.reply(405)

                do_DELETE = do_GET

            self.server = HTTPServer(("127.0.0.1", 0), Handler)
            self.server.timeout = 1
            profile.add_server("agsci_python", {
                "command": sys.executable,
                "args": [str(self.installed["bridge.py"]), "--url",
                         f"http://127.0.0.1:{self.server.server_port}/mcp", "--token-file", str(self.token_path)],
                "required": True, "startup_timeout_sec": 20, "tool_timeout_sec": 60,
                "enabled_tools": ["get_execution_policy", "run_python"],
                "default_tools_approval_mode": "prompt",
                "tools": {name: {"approval_mode": "approve"} for name in ["get_execution_policy", "run_python"]},
            })
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True,
                                           name="agsci-python-" + self.broker.scope)
            self.thread.start()
        except BaseException:
            if hasattr(self, "server"):
                self.server.server_close()
            if hasattr(self, "token_path"):
                self.token_path.unlink(missing_ok=True)
            self.broker.close()
            raise

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        if method == "initialize":
            self.initialized = True
            return {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "agsci-isolated-python", "version": "1"}}
        if method == "tools/list" and self.initialized:
            self.listed = True
            return {"tools": tools()}
        if method == "ping":
            return {}
        try:
            if method != "tools/call" or not self.listed:
                raise ValueError("Initialize and list Python tools before calling them")
            params = request.get("params", {})
            if not isinstance(params, dict) or not isinstance(params.get("arguments", {}), dict):
                raise ValueError("Tool arguments must be an object")
            if params.get("name") == "get_execution_policy":
                if params.get("arguments", {}):
                    raise ValueError("Execution policy takes no arguments")
                return _tool_result({"policy": self.broker.policy, "policy_sha256": self.broker.policy_sha256})
            if params.get("name") != "run_python":
                raise ValueError("Unknown Python tool")
            return _tool_result(self.broker.execute(params.get("arguments", {})))
        except (ValueError, OSError) as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}

    def close(self) -> None:
        if self.closed:
            return
        self.stop_event.set()
        shutdown = threading.Thread(target=self.server.shutdown, daemon=True)
        shutdown.start()
        self.thread.join(timeout=55)
        if self.thread.is_alive():
            raise ValueError("Python execution service has not stopped; recovery lease retained")
        shutdown.join(timeout=1)
        self.server.server_close()
        self.token_path.unlink(missing_ok=True)
        self.broker.close()
        self.closed = True

    def finish(self) -> dict[str, Any]:
        self.close()
        if self.broker.broken:
            raise ValueError("Python execution cleanup was not verified")
        for filename, installed in self.installed.items():
            if _sha(_regular(installed, MAX_AUDIT_BYTES)) != self.module_hashes[filename]:
                raise ValueError("Python execution module changed during invocation")
        binding = {"schema_version": 1, "policy": self.broker.policy,
                   "policy_sha256": self.broker.policy_sha256, "prompt_sha256": self.broker.prompt_sha256,
                   "module_sha256": self.module_hashes, "audit_sha256": _sha(_regular(self.audit_path, MAX_AUDIT_BYTES)),
                   "profile_source_sha256": self.profile.source_sha256,
                   "private_profile_sha256": self.profile.verify(),
                   "recovered_jobs": self.broker.recovered_jobs,
                   "jobs": [job["name"] for job in self.broker.state["jobs"]],
                   "service_shutdown_verified": self.closed, "ephemeral_token_deleted": not self.token_path.exists(),
                   "credential_files_exported": False, "semantic_validity_verified": False}
        body = (_json(binding) + "\n").encode()
        _private(self.root / "binding.json", body)
        receipt = {"policy": POLICY, "binding_path": DIRECTORY + "/binding.json", "binding_sha256": _sha(body)}
        verify_execution_tools(self.root.parent, receipt, expected_prompt_sha256=self.broker.prompt_sha256)
        return receipt


def verify_execution_tools(artifacts: Path, receipt: dict[str, Any], *, expected_prompt_sha256: str) -> dict[str, Any]:
    """Verify saved execution bytes and bindings; never execute model code here."""
    if (not isinstance(receipt, dict) or receipt.get("policy") != POLICY
            or receipt.get("binding_path") != DIRECTORY + "/binding.json"):
        raise ValueError("Invalid Python execution receipt")
    root = artifacts / DIRECTORY
    body = _regular(root / "binding.json", MAX_AUDIT_BYTES)
    if _sha(body) != receipt.get("binding_sha256"):
        raise ValueError("Python execution binding changed")
    binding = _parse(body.decode())
    policy = binding.get("policy", {})
    if (binding.get("schema_version") != 1 or policy != execution_policy(policy.get("image"))
            or binding.get("policy_sha256") != _sha(_json(policy))
            or binding.get("prompt_sha256") != expected_prompt_sha256
            or binding.get("service_shutdown_verified") is not True
            or binding.get("ephemeral_token_deleted") is not True
            or binding.get("credential_files_exported") is not False
            or binding.get("semantic_validity_verified") is not False):
        raise ValueError("Python execution policy, prompt or cleanup binding changed")
    for name in ["runner.py", "bridge.py", "adapter.py"]:
        if _sha(_regular(root / name, MAX_AUDIT_BYTES)) != binding.get("module_sha256", {}).get(name):
            raise ValueError("Python execution module snapshot changed")
    jobs = binding.get("jobs")
    if (not isinstance(jobs, list) or len(jobs) > 12 or len(set(jobs)) != len(jobs)
            or not all(isinstance(n, str) and re.fullmatch(r"agsci-python-[0-9a-f]{32}", n) for n in jobs)):
        raise ValueError("Invalid Python job list")
    results = {}
    for name in jobs:
        path = root / name
        raw = _regular(path / "receipt.json", 65536)
        saved = _parse(raw.decode())
        out, err = _regular(path / "stdout.bin", 65536), _regular(path / "stderr.bin", 65536)
        code, data = _regular(path / "program.py", 32768), _regular(path / "input.json", 65536)
        if (saved.get("schema_version") != 1 or saved.get("job_id") != name
                or saved.get("prompt_sha256") != expected_prompt_sha256
                or saved.get("policy_sha256") != binding["policy_sha256"]
                or saved.get("cleanup_verified") is not True
                or saved.get("status") not in {"completed", "program_failed", "timeout", "output_limit", "cancelled", "infrastructure_error"}
                or saved.get("stdout_sha256") != _sha(out) or saved.get("stderr_sha256") != _sha(err)
                or saved.get("code_sha256") != _sha(code) or saved.get("input_sha256") != _sha(data)):
            raise ValueError("Python program, input, output or cleanup receipt changed")
        if saved["status"] == "completed" and (saved.get("exit_code") != 0
                or saved.get("stdout_truncated") is not False or saved.get("stderr_truncated") is not False):
            raise ValueError("Completed Python job is incomplete")
        if saved["status"] in {"completed", "program_failed", "timeout", "output_limit", "cancelled"}:
            isolation_bytes = _regular(path / "isolation.json", 65536)
            if _sha(isolation_bytes) != saved.get("isolation_sha256"):
                raise ValueError("Python container isolation record changed")
            isolation = _parse(isolation_bytes.decode())
            if (isolation.get("container_id") != saved.get("container_id")
                    or isolation.get("image") != policy["image"] or isolation.get("network") != "none"
                    or isolation.get("read_only_root") is not True or isolation.get("user") != "1000:1000"
                    or isolation.get("memory_bytes") != policy["memory_bytes"]
                    or isolation.get("memory_swap_bytes") != policy["memory_bytes"]
                    or isolation.get("cpus") != 1000000000 or isolation.get("pids") != 64
                    or isolation.get("cap_drop") != ["ALL"] or isolation.get("gpu_requests")):
                raise ValueError("Python job isolation differs from policy")
        results[name] = {**saved, "receipt_sha256": _sha(raw),
                         "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace")}
    audit = _regular(root / "calls.jsonl", MAX_AUDIT_BYTES)
    if _sha(audit) != binding.get("audit_sha256"):
        raise ValueError("Python execution audit changed")
    initialized = listed = False
    seen = []
    for line in audit.decode().splitlines():
        event = _parse(line)
        request, result = event.get("request", {}), event.get("result", {})
        if (event.get("prompt_sha256") != expected_prompt_sha256
                or event.get("request_sha256") != _sha(_json(request))
                or event.get("result_sha256") != _sha(_json(result))):
            raise ValueError("Python MCP request or result changed")
        method = request.get("method")
        if method == "initialize":
            initialized = result == {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "agsci-isolated-python", "version": "1"}}
        elif method == "tools/list":
            listed = initialized and result == {"tools": tools()}
        elif method == "tools/call":
            if not listed:
                raise ValueError("Python tool called before catalog delivery")
            params = request.get("params", {})
            if params.get("name") == "get_execution_policy" and result.get("isError") is False:
                if result != _tool_result({"policy": policy, "policy_sha256": binding["policy_sha256"]}):
                    raise ValueError("Python policy tool result changed")
            elif params.get("name") == "run_python":
                text = result.get("content", [{}])[0].get("text", "")
                try:
                    value = _parse(text)
                except ValueError:
                    if result.get("isError") is not True:
                        raise ValueError("Python tool has no bound execution result") from None
                    continue
                name = value.get("job_id")
                if name not in results or result != _tool_result(results[name]) or name in seen:
                    raise ValueError("Python MCP result differs from the actual execution receipt")
                args = params.get("arguments", {})
                if (_sha(args.get("code", "")) != results[name]["code_sha256"]
                        or _sha(_json(args.get("input", {}))) != results[name]["input_sha256"]):
                    raise ValueError("Python MCP request differs from the executed program or input")
                seen.append(name)
    if not (initialized and listed) or seen != jobs:
        raise ValueError("Python catalog or execution calls were not fully delivered")
    return {**binding, "tool_catalog_delivered": True, "execution_jobs_verified": len(jobs)}
