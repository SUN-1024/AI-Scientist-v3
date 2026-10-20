"""Host-owned, bounded Python jobs. Model code runs only inside Docker."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any
import uuid

from src.final_evaluation.broker_storage import broker_host_path, broker_root, ensure_broker_root
from src.investigation.bounded_process import (
    BoundedProcessError, ProcessCancelledError, ProcessOutputLimitError,
    ProcessTimeoutError, run_bounded_process,
)
from .agent_tool_profile import _private, _regular
from .source_lookup import _json, _parse, _sha

POLICY = "isolated_python_v1"
MAX_AUDIT_BYTES = 8 * 1024 * 1024
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX = re.compile(r"[0-9a-f]{32}\Z")
_CID = re.compile(r"[0-9a-f]{64}\Z")


def execution_policy(image: str) -> dict[str, Any]:
    if not isinstance(image, str) or not _IMAGE.fullmatch(image):
        raise ValueError("Python execution requires an immutable sha256 Docker image ID")
    return {"schema_version": 1, "policy": POLICY, "image": image,
            "network": "none", "user": "1000:1000", "cpus": 1,
            "memory_bytes": 536870912, "pids": 64, "timeout_seconds": 10,
            "output_bytes": 65536, "max_code_bytes": 32768, "max_input_bytes": 65536,
            "maximum_calls": 12, "gpu": False, "host_credentials_mounted": False,
            "arbitrary_host_commands_allowed": False}


def _save(path: Path, value: dict[str, Any]) -> None:
    body = (_json(value) + "\n").encode()
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex)
    _private(temporary, body)
    try:
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class FinitePythonBroker:
    """One leased invocation; persistent ownership records survive interruption."""

    def __init__(self, artifacts: Path, *, image: str, prompt_sha256: str, cancel_event=None):
        self.policy = execution_policy(image)
        if not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256):
            raise ValueError("Python execution must bind to an exact prompt")
        self.policy_sha256 = _sha(_json(self.policy))
        self.prompt_sha256 = prompt_sha256
        self.artifacts = artifacts
        self.calls = 0
        self.broken = False
        self.cancel_event = cancel_event
        self.scope = _sha(str(artifacts.absolute()))[:32]
        self.root = ensure_broker_root(broker_root()) / ("finite-python-" + self.scope)
        self.root.mkdir(mode=0o700, exist_ok=True)
        info = self.root.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("Unsafe Python execution recovery directory")
        self.env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(self.root)}
        self.lock_fd = os.open(self.root / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.state_path = self.root / "session.json"
        self.state: dict[str, Any] = {}
        self.recovered_jobs = 0
        try:
            meta = os.fstat(self.lock_fd)
            if (not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid()
                    or stat.S_IMODE(meta.st_mode) != 0o600):
                raise ValueError("Unsafe Python execution owner lease")
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.state_path.exists() or self.state_path.is_symlink():
                previous = _parse(_regular(self.state_path, 1024 * 1024).decode())
                self._validate_state(previous)
                self.state = previous
                for job in previous["jobs"]:
                    if not job["cleaned"]:
                        self._cleanup(job)
                        self.recovered_jobs += 1
                self.state_path.unlink()
            self.state = {"schema_version": 1, "scope": self.scope,
                          "session": uuid.uuid4().hex, "jobs": []}
            probe = self.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
            if probe.returncode or probe.stdout.decode().strip() != image:
                raise ValueError("Pinned Python execution image is unavailable")
            artifacts.mkdir()
            _save(self.state_path, self.state)
        except BaseException:
            os.close(self.lock_fd)
            self.lock_fd = -1
            raise

    def _validate_state(self, state: dict[str, Any]) -> None:
        if (state.get("schema_version") != 1 or state.get("scope") != self.scope
                or not isinstance(state.get("session"), str) or not _HEX.fullmatch(state["session"])
                or not isinstance(state.get("jobs"), list) or len(state["jobs"]) > 12):
            raise ValueError("Invalid Python execution recovery record")
        names = set()
        for job in state["jobs"]:
            if (not isinstance(job, dict) or not isinstance(job.get("name"), str)
                    or not re.fullmatch(r"agsci-python-[0-9a-f]{32}", job["name"])
                    or job["name"] in names or type(job.get("cleaned")) is not bool
                    or (job.get("cid") is not None and (not isinstance(job["cid"], str)
                                                       or not _CID.fullmatch(job["cid"])))):
                raise ValueError("Invalid Python job recovery identity")
            names.add(job["name"])

    def run(self, argv, *, timeout=15, output=65536, cancellable=False):
        return run_bounded_process(
            argv, cwd=self.root, env=self.env, timeout_seconds=timeout, max_output_bytes=output,
            cancel_event=self.cancel_event if cancellable else None,
        )

    def _inspect(self, identity: str) -> dict[str, Any] | None:
        result = self.run(["docker", "inspect", identity])
        if result.returncode:
            error = result.stderr.decode(errors="replace")
            if "No such object:" in error or "No such container:" in error:
                return None
            raise ValueError("Cannot verify Python container state")
        rows = json.loads(result.stdout)
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("Invalid Python container inspection")
        return rows[0]

    def _owned(self, observed: dict[str, Any], job: dict[str, Any]) -> None:
        labels = observed.get("Config", {}).get("Labels", {})
        if (not isinstance(observed.get("Id"), str) or not _CID.fullmatch(observed["Id"])
                or observed.get("Name") != "/" + job["name"]
                or labels.get("agsci.finite.scope") != self.scope
                or labels.get("agsci.finite.session") != self.state["session"]
                or (job["cid"] is not None and observed.get("Id") != job["cid"])):
            raise ValueError("Python container ownership differs; no container removed")

    def _cleanup(self, job: dict[str, Any]) -> None:
        observed = self._inspect(job["cid"] or job["name"])
        if observed is not None:
            self._owned(observed, job)
            result = self.run(["docker", "rm", "--force", observed["Id"]])
            if result.returncode or self._inspect(observed["Id"]) is not None:
                raise ValueError("Python container cleanup could not be verified")
        elif job["cid"] is None:
            # A timed-out create may still reach the daemon after its client
            # exits. Retain ownership metadata rather than certify absence.
            raise ValueError("Python creation was not acknowledged; recovery metadata retained")
        job["cleaned"] = True
        _save(self.state_path, self.state)
        inputs = self.root / job["name"]
        if inputs.exists():
            if inputs.is_symlink():
                raise ValueError("Python recovery input directory changed")
            shutil.rmtree(inputs)

    def _isolation(self, observed: dict[str, Any], job: dict[str, Any], host_input: Path) -> dict[str, Any]:
        self._owned(observed, job)
        config, host = observed["Config"], observed["HostConfig"]
        mounts = observed["Mounts"]
        binds = [m for m in mounts if m["Type"] != "tmpfs"]
        if not (observed["Image"] == self.policy["image"] and host["ReadonlyRootfs"]
                and host["NetworkMode"] == "none" and host["CapDrop"] == ["ALL"]
                and not host["Privileged"] and not host.get("DeviceRequests")
                and not host.get("Devices") and not host.get("Binds")
                and "no-new-privileges" in host["SecurityOpt"]
                and host["Memory"] == host["MemorySwap"] == 536870912
                and host["NanoCpus"] == 1000000000 and host["PidsLimit"] == 64
                and config["User"] == "1000:1000" and len(binds) == 1
                and binds[0]["Type"] == "bind" and not binds[0]["RW"]
                and binds[0]["Destination"] == "/input" and binds[0]["Source"] == str(host_input)):
            raise ValueError("Python container isolation differs from its policy")
        return {"image": observed["Image"], "container_id": observed["Id"],
                "user": config["User"], "network": host["NetworkMode"],
                "read_only_root": host["ReadonlyRootfs"], "memory_bytes": host["Memory"],
                "memory_swap_bytes": host["MemorySwap"], "cpus": host["NanoCpus"],
                "pids": host["PidsLimit"], "cap_drop": host["CapDrop"],
                "security_opt": host["SecurityOpt"], "mounts": mounts, "gpu_requests": host.get("DeviceRequests")}

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.broken or self.calls >= self.policy["maximum_calls"]:
            raise ValueError("Python execution unavailable or call budget exhausted")
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ValueError("Python execution cancelled")
        if not isinstance(arguments, dict) or set(arguments) - {"code", "input"}:
            raise ValueError("Python execution supports only code and input")
        code, data = arguments.get("code"), arguments.get("input", {})
        if not isinstance(code, str) or not code.strip() or len(code.encode()) > 32768:
            raise ValueError("Python source must be nonempty and at most 32768 UTF-8 bytes")
        if not isinstance(data, dict) or len(_json(data).encode()) > 65536:
            raise ValueError("Python input must be a JSON object of at most 65536 bytes")
        self.calls += 1
        name = "agsci-python-" + uuid.uuid4().hex
        evidence = self.artifacts / name
        evidence.mkdir(mode=0o700)
        _private(evidence / "program.py", code.encode())
        _private(evidence / "input.json", _json(data).encode())
        private_job = self.root / name
        private_job.mkdir(mode=0o700)
        host_job = broker_host_path(private_job)
        inputs = private_job / "input"
        inputs.mkdir(mode=0o755)
        inputs.chmod(0o755)
        for filename in ["program.py", "input.json"]:
            _private(inputs / filename, _regular(evidence / filename, 65536))
            (inputs / filename).chmod(0o444)
        job = {"name": name, "cid": None, "cleaned": False}
        self.state["jobs"].append(job)
        _save(self.state_path, self.state)
        receipt: dict[str, Any] = {
            "schema_version": 1, "job_id": name, "status": "started",
            "prompt_sha256": self.prompt_sha256, "policy_sha256": self.policy_sha256,
            "code_sha256": _sha(code), "input_sha256": _sha(_json(data)),
            "at": datetime.now(timezone.utc).isoformat(), "cleanup_verified": False,
            "stdout_truncated": False, "stderr_truncated": False,
        }
        _save(evidence / "receipt.json", receipt)
        stdout, stderr = b"", b""
        try:
            bootstrap = ("import json;from pathlib import Path;"
                         "exec(compile(Path('/input/program.py').read_text(),'/input/program.py','exec'),"
                         "{'__name__':'__main__','INPUT_DATA':json.loads(Path('/input/input.json').read_text())})")
            created = self.run([
                "docker", "create", "--pull=never", "--name", name,
                "--label", "agsci.finite.scope=" + self.scope,
                "--label", "agsci.finite.session=" + self.state["session"],
                "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "64", "--cpus", "1",
                "--memory", "512m", "--memory-swap", "512m", "--user", "1000:1000",
                "--log-driver", "none", "--init", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m",
                "--tmpfs", "/work:rw,nosuid,nodev,noexec,size=32m,uid=1000,gid=1000,mode=700",
                "--workdir", "/work", "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
                "--mount", f"type=bind,src={host_job / 'input'},dst=/input,readonly",
                "--entrypoint", "/opt/research-runner/venv/bin/python", self.policy["image"],
                "-I", "-B", "-c", bootstrap,
            ])
            cid = created.stdout.decode().strip()
            if created.returncode or not _CID.fullmatch(cid):
                raise ValueError("Python container creation was not acknowledged")
            job["cid"] = cid
            _save(self.state_path, self.state)
            receipt["container_id"] = cid
            observed = self._inspect(cid)
            if observed is None:
                raise ValueError("Python container disappeared before validation")
            _save(evidence / "isolation.json", self._isolation(observed, job, host_job / "input"))
            receipt["isolation_sha256"] = _sha(_regular(evidence / "isolation.json", 65536))
            result = self.run(["docker", "start", "--attach", cid], timeout=10, cancellable=True)
            stdout, stderr = result.stdout, result.stderr
            final = self._inspect(cid)
            if final is None or final["State"]["Running"]:
                raise ValueError("Cannot verify terminal Python container state")
            self._owned(final, job)
            state = final["State"]
            if result.returncode and state["ExitCode"] == 0:
                raise ValueError("Python attach failed despite a zero program exit")
            receipt.update(status="completed" if state["ExitCode"] == 0 else "program_failed",
                           exit_code=state["ExitCode"], oom_killed=state["OOMKilled"])
        except (ProcessTimeoutError, ProcessOutputLimitError) as exc:
            stdout, stderr = exc.stdout, exc.stderr
            receipt.update(status="timeout" if isinstance(exc, ProcessTimeoutError) else "output_limit",
                           stdout_truncated=exc.stdout_truncated, stderr_truncated=exc.stderr_truncated)
        except ProcessCancelledError:
            receipt.update(status="cancelled", stdout_truncated=True, stderr_truncated=True)
        except (ValueError, OSError, BoundedProcessError) as exc:
            receipt.update(status="infrastructure_error", error=type(exc).__name__ + ": " + str(exc)[:500])
        finally:
            try:
                self._cleanup(job)
                receipt["cleanup_verified"] = True
            except (ValueError, OSError, BoundedProcessError):
                self.broken = True
                receipt["status"] = "cleanup_unverified"
            _private(evidence / "stdout.bin", stdout)
            _private(evidence / "stderr.bin", stderr)
            receipt.update(stdout_sha256=_sha(stdout), stderr_sha256=_sha(stderr),
                           finished_at=datetime.now(timezone.utc).isoformat())
            _save(evidence / "receipt.json", receipt)
        return {**receipt, "receipt_sha256": _sha(_regular(evidence / "receipt.json", 65536)),
                "stdout": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace")}

    def close(self) -> None:
        if self.lock_fd < 0:
            return
        try:
            for job in self.state.get("jobs", []):
                if not job["cleaned"]:
                    self._cleanup(job)
            self.state_path.unlink(missing_ok=True)
        finally:
            os.close(self.lock_fd)
            self.lock_fd = -1
