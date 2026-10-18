"""Private broker storage with short local paths and daemon-visible bind paths."""
from pathlib import Path
import os
import stat


def _default_broker_root() -> Path:
    return Path("/tmp") / f"agentic-scientist-broker-{os.getuid()}"


def _configured_roots() -> tuple[Path, Path] | None:
    names = ("AGENTIC_BROKER_ROOT", "AGENTIC_BROKER_HOST_ROOT")
    values = tuple(os.environ.get(name) for name in names)
    if values == (None, None):
        return None
    roots = []
    for value in values:
        if (not value or not value.startswith("/") or value.startswith("//")
                or any(part in ("", ".", "..") for part in value.split("/")[1:])
                or any(character in value for character in ",\r\n\x00")):
            raise ValueError("broker storage requires two safe absolute roots")
        roots.append(Path(value))
    local, host = roots
    if local.name != host.name:
        raise ValueError("broker storage aliases must have the same directory name")
    # Leave room for draft-search-<32 hex> and evaluator-start.sock on macOS
    # (103 bytes) as well as Linux (107 bytes). Do not resolve the short alias.
    longest_address = local / ("draft-search-" + "0" * 32) / "evaluator-start.sock"
    if len(os.fsencode(longest_address)) > 103:
        raise ValueError("broker local root is too long for Unix sockets")
    return local, host


def broker_root() -> Path:
    configured = _configured_roots()
    return configured[0] if configured is not None else _default_broker_root()


def legacy_broker_root() -> Path:
    return Path("/dev/shm") / f"agentic-scientist-broker-{os.getuid()}"


def _validate_root(root: Path) -> None:
    metadata = root.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise ValueError("unsafe broker recovery root")


def _validate_parent_chain(parent: Path) -> None:
    """Reject replaceable ancestors and symlinks before trusting an alias."""
    _validate_root(parent)
    for ancestor in parent.parents:
        metadata = ancestor.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in (0, os.getuid())
                or (metadata.st_mode & 0o022
                    and not (metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX))):
            raise ValueError("unsafe broker storage ancestor")


def _validate_aliases(local: Path, host: Path) -> None:
    for parent in {local.parent, host.parent}:
        _validate_parent_chain(parent)
    if not os.path.samestat(local.parent.stat(), host.parent.stat()):
        raise ValueError("broker storage aliases do not refer to the same directory")


def ensure_broker_root(root: Path) -> Path:
    configured = _configured_roots()
    if configured is not None:
        local, host = configured
        if root != local:
            raise ValueError("broker root differs from the configured storage")
        _validate_aliases(local, host)
    else:
        parent = Path("/tmp").lstat()
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0
                or (parent.st_mode & 0o022 and not parent.st_mode & stat.S_ISVTX)):
            raise ValueError("unsafe broker temporary directory")
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _validate_root(root)
    return root


def broker_host_path(path: Path) -> Path:
    """Translate only broker bind sources; Docker cidfiles are client-local."""
    configured = _configured_roots()
    if configured is None:
        return path
    local, host = configured
    _validate_aliases(local, host)
    _validate_root(local)
    if ".." in path.parts:
        raise ValueError("broker bind path escapes its configured root")
    try:
        relative = path.relative_to(local)
    except ValueError as exc:
        raise ValueError("broker bind path escapes its configured root") from exc
    current = local
    for part in relative.parts:
        if any(character in part for character in ",\r\n\x00"):
            raise ValueError("unsafe broker bind path")
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue  # Commands are assembled before CandidateBroker creates it.
        if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700):
            raise ValueError("unsafe broker bind directory")
    return host / relative


def reject_legacy_broker_directory(directory: Path) -> None:
    """Never abandon an interrupted old runtime when adopting the new location."""
    previous_roots = [legacy_broker_root()]
    if _configured_roots() is not None and directory.parent != _default_broker_root():
        previous_roots.append(_default_broker_root())
    for legacy_root in previous_roots:
        try:
            _validate_root(legacy_root)
        except FileNotFoundError:
            continue
        legacy = legacy_root / directory.name
        try:
            legacy.lstat()
        except FileNotFoundError:
            continue
        raise ValueError(
            f"legacy broker recovery metadata remains at {legacy}; "
            "verify and clean up the old runtime before retrying"
        )
