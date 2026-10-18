# Linux Harness runner

This image runs the trusted Python orchestrator on Linux. Linux supplies the
`/proc/self/fd` directory operations used by the sealed evaluator. Durable run
state uses a dedicated Docker volume. Docker Desktop's macOS shared mounts do
not provide all the rename, inode, and permission behavior required by paper
publication and checkpoints. The runner preserves those checks.

The runner uses the CPU research image for Codex, Headless, Node, and TeX. It adds
the Docker client, pinned `uv` for legacy standalone review, and a separate Python environment installed from the
repository's `uv.lock`. The export includes development dependencies so the same
image can run the repository tests. The mounted checkout supplies current source.
The macOS `.venv` is never used.

## Build and check

From the checkout:

```bash
scripts/build-research-cpu-image.sh
scripts/build-research-runner-image.sh
scripts/run-research-harness-container.sh --help
export RESEARCH_RUNNER_VOLUME=agsci-unified-research-state
scripts/run-research-harness-container.sh --native-state --state-info
scripts/run-research-harness-container.sh --native-state --preflight
```

`--preflight` verifies Python, dependencies, the Git revision, Docker client and
daemon, the CPU image, Codex, Headless, actual TeX compilation, Linux directory
pinning, atomic directory exchange, private file permissions, the same sentinel
inode from a sibling container, and the provider's `/models` endpoint. It does
not invoke an LLM or train a model. Additional arguments select images to inspect.

Builds use the Aliyun package mirrors configured in the CPU image. The wrapper
uses a temporary Docker client config with no ambient proxy or credentials.
An explicitly supplied Docker build argument can override a package mirror or
set a proxy reachable from the build container. It never modifies host settings.

## Run a case

Supply the same arguments that normally follow `agsci`. Use the exact
`native_root` returned by `--state-info`; Docker's data directory can vary:

```bash
export RESEARCH_RUNNER_VOLUME=agsci-unified-research-state
native_root=/var/lib/docker/volumes/agsci-unified-research-state/_data
scripts/run-research-harness-container.sh --native-state research harness \
  examples/pathmnist_mil_harness --offline --dry-run \
  --workspace "$native_root/workspaces/pathmnist-development" \
  container.image=unified-ai-scientist/research-cpu:2026.10
```

Remove `--dry-run` only when model calls and execution are intended. The wrapper
enables native state by default for complete macOS `research harness` runs.
`--workspace` and `--resume` must name a path within the selected volume's
`workspaces/` directory; the wrapper rejects another volume before model calls.
The source task can remain in the read-only checkout. The Harness copies it into
the native study workspace before execution. Task environments are created there.
To run tests on that same filesystem:

```bash
scripts/run-research-harness-container.sh --native-state --exec \
  python -m pytest tests/research/test_harness.py -q -p no:cacheprovider
```

The checkout and its common Git metadata are mounted at their **same absolute
paths, read-only** in native mode. The volume is mounted at the daemon's exact
`Mountpoint`, not an arbitrary container alias. A sibling task container can
then bind the same source path and see the same files. The volume holds active
HOME, temporary files, studies, managed environments, and checkpoint repositories.
No writable runtime state needs to be placed on the macOS shared filesystem.

The coordinator also mounts that volume at `/agsci-state` for short Unix socket
addresses. `AGENTIC_BROKER_ROOT` names its private `broker-<uid>` directory;
`AGENTIC_BROKER_HOST_ROOT` names the same directory at the daemon's canonical
volume path. Broker Docker bind sources use the latter. Socket and cidfile paths
use the former. Both aliases must have the same private parent inode, with no
symlinks or unsafe ancestors. The wrapper sets both variables together. Without
these variables, host execution keeps its existing private `/tmp` broker root.
Non-native mode supplies the same alias mapping from private launcher storage;
complete macOS runs still require native state for filesystem guarantees.

Each invocation creates a private launcher directory under
`scratch/research-runner/`. Native mode then creates its active HOME and Docker
client config under the volume's `sessions/`. Only the named Codex profile is
copied from `RESEARCH_RUNNER_AUTH_ROOT/.codex/`; the default source is
`scratch/harness-audit/container-auth`. The default profile name is `agsci-magpie`.
The host user's global Codex home is not mounted. The provider URL in this profile
must be reachable from Linux, such as `http://host.docker.internal:3425/v1` for the
local Magpie bridge. Desktop-only MCP entries should be removed from this private
profile before use.

For an external literature or review provider, explicitly set
`RESEARCH_RUNNER_ENV_FILE=/private/path/outer.env`. Docker supplies that file's
variables only to the trusted outer orchestrator. The wrapper does not copy the
file, log its contents, or inherit all host API keys. `container.env_file` is a
separate pipeline setting for inner coding-agent containers. The Magpie profile
used here requires no API key file.

After a study stops, export an inspection snapshot to a **new** host directory:

```bash
scripts/run-research-harness-container.sh --native-state \
  --export-workspace pathmnist-development results/exports/pathmnist-development
```

Export only reads the selected directory below `workspaces/`. It does not export
volume HOME or sessions. It omits `.venv`, agent auth directories, `.env` files,
and caches. Study-local checkpoint sidecars remain included. It preserves
symlinks without following them, keeps sealed receipts and native path references
unchanged, and removes write permission from the resulting snapshot. The
`runner-export.json` records the original native path and omissions. Resume only
from the native workspace. An export never becomes the active run state. The
wrapper does not delete volumes or previous studies.

The outer orchestrator has Docker socket access because it starts sealed child
containers. Docker socket access grants control over the daemon. The runner does
not pass its socket into coding-agent containers. Existing explicit inner-runtime
settings remain responsible for any such access. Outputs use the host user's UID
and GID. Docker Desktop additionally requires supplemental group `0` for its
socket; this does not run the orchestrator as root.

Supported environment settings:

| Variable | Purpose |
| --- | --- |
| `RESEARCH_RUNNER_IMAGE` | Runner image, default `unified-ai-scientist/research-runner:2026.10` |
| `RESEARCH_CPU_IMAGE` | CPU base image and default preflight target |
| `RESEARCH_RUNNER_AUTH_ROOT` | Directory containing the private `.codex/<profile>.config.toml` |
| `RESEARCH_RUNNER_CODEX_PROFILE` | Named profile, default `agsci-magpie` |
| `RESEARCH_RUNNER_NATIVE_STATE` | Set to `1` to select native state without the flag |
| `RESEARCH_RUNNER_VOLUME` | Dedicated volume name; otherwise derived from the checkout path |
| `RESEARCH_RUNNER_STATE_DIR` | Private launcher state within this checkout |
| `RESEARCH_RUNNER_ENV_FILE` | Explicit outer provider environment; never copied into snapshots |
| `AGENTIC_TEMP_ROOT` | In non-native mode only, a temporary directory within this checkout |
| `AGENTIC_TASK_PACKAGE_INDEX` | Task package mirror; default Aliyun PyPI |
| `RESEARCH_DOCKER_SOCKET_GID` | Socket group override; auto-detected on Linux, `0` on Docker Desktop for macOS |
| `DOCKER_BIN` / `DOCKER_HOST` | Docker binary and local Unix socket endpoint |

The wrapper supports local Unix sockets. It rejects remote Docker endpoints
because a remote daemon cannot resolve the local source paths used here.
No package installation, Docker configuration change, or service restart occurs
on the host.
