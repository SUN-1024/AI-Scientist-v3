"""Bounded, exact-byte review inputs shared by the host and OpenAI reviewer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from src.investigation.pinned_files import (
    absolute_path, open_directory_tree_no_follow, read_regular_at,
)

PAPER_PDF_MAX_BYTES = 100 * 1024 * 1024
REVIEW_EVIDENCE_MAX_BYTES = 2 * 1024 * 1024


def read_review_input(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded input without following any directory or file symlink."""
    path = absolute_path(path)
    directory_fd = open_directory_tree_no_follow(path.parent)
    try:
        return read_regular_at(
            directory_fd, path.name, max_bytes=max_bytes, description="review input"
        )
    finally:
        os.close(directory_fd)


def review_input_record(content: bytes, *, source_id: str, name: str) -> dict[str, object]:
    return {
        "id": source_id, "name": name,
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
    }
