"""Credential-file stdio bridge to one invocation's loopback MCP endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}/mcp", args.url):
        raise ValueError("Execution bridge requires a fixed loopback endpoint")
    token = args.token_file.read_text()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    while True:
        raw = sys.stdin.buffer.readline(131073)
        if not raw:
            return 0
        if len(raw) > 131072:
            raise ValueError("MCP request size exceeded")
        request = json.loads(raw)
        try:
            headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json",
                       "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
            with opener.open(urllib.request.Request(args.url, data=raw, headers=headers), timeout=60) as response:
                payload = response.read(524289)
            if len(payload) > 524288:
                raise ValueError("MCP response size exceeded")
            if payload:
                sys.stdout.write(json.dumps(json.loads(payload), ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except (OSError, ValueError) as exc:
            if "id" in request:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request["id"],
                    "error": {"code": -32603, "message": "Execution bridge failed: " + type(exc).__name__}}) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
