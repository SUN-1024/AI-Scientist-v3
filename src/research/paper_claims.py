"""Claim rules for manuscripts: required and forbidden phrases.

Review loops repeatedly catch the same classes of wording: a draft that still calls itself
submission-ready, cites a stale hash, or claims peer review. This module checks a manuscript
against a small JSON rule set so those red lines are enforced mechanically instead of by memory.

Rule file format::

    {"required": [{"pattern": "manifest SHA-256", "reason": "bundle citation"}],
     "forbidden": [{"pattern": "submission-ready", "reason": "no such state is claimed"}]}

Patterns are Python regular expressions matched case-insensitively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def check_claims(text: str, rules: dict[str, Any]) -> dict[str, Any]:
    results = {"required": [], "forbidden": [], "status": "passed"}
    for rule in rules.get("required", []):
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        found = bool(pattern.search(text))
        results["required"].append({"pattern": rule["pattern"], "reason": rule.get("reason", ""),
                                    "found": found})
        if not found:
            results["status"] = "failed"
    for rule in rules.get("forbidden", []):
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        unless = re.compile(rule["unless"], re.IGNORECASE) if rule.get("unless") else None
        matches = 0
        for match in pattern.finditer(text):
            if unless is not None:
                start = text.rfind("\n\n", 0, match.start()) + 2
                end = text.find("\n\n", match.end())
                paragraph = text[start:end if end != -1 else len(text)]
                if unless.search(paragraph):
                    continue
            matches += 1
        results["forbidden"].append({"pattern": rule["pattern"], "reason": rule.get("reason", ""),
                                     "unless": rule.get("unless"), "matches": matches})
        if matches:
            results["status"] = "failed"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = check_claims(args.manuscript.read_text(), json.loads(args.rules.read_text()))
    report["manuscript"] = str(args.manuscript)
    body = json.dumps(report, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(body + "\n")
    print(body)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
