#!/usr/bin/env bash
# Run the trusted orchestrator on Linux while child tasks use the host daemon.
set -euo pipefail
umask 077
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
native="${RESEARCH_RUNNER_NATIVE_STATE:-0}"
if [ "${1:-}" = --native-state ]; then native=1; shift; fi
# Complete Harness runs must not put durable state on a macOS shared mount.
if [ "$(uname -s)" = Darwin ] && [ "${1:-}" = research ] && [ "${2:-}" = harness ]; then
  dry_run=0
  for argument in "$@"; do
    case "$argument" in --dry-run|execution.dry_run=true) dry_run=1 ;; esac
  done
  if [ "$dry_run" = 0 ]; then native=1; fi
fi
image="${RESEARCH_RUNNER_IMAGE:-unified-ai-scientist/research-runner:2026.10}"
docker_bin="${DOCKER_BIN:-docker}"
if ! command -v "$docker_bin" >/dev/null 2>&1; then
  docker_bin=/Applications/Docker.app/Contents/Resources/bin/docker
fi
command -v "$docker_bin" >/dev/null 2>&1 || { echo 'Docker is required.' >&2; exit 2; }
endpoint="${DOCKER_HOST:-$("$docker_bin" context inspect --format '{{.Endpoints.docker.Host}}')}"
case "$endpoint" in
  unix://*) socket="${endpoint#unix://}" ;;
  *) echo 'This runner requires a local Unix Docker socket.' >&2; exit 2 ;;
esac
test -S "$socket" || { echo "Docker socket is missing: $socket" >&2; exit 2; }
state="${RESEARCH_RUNNER_STATE_DIR:-$root/scratch/research-runner}"
mkdir -p "$state"
state="$(cd "$state" && pwd -P)"
case "$state/" in "$root/"*) ;; *) echo 'RESEARCH_RUNNER_STATE_DIR must be inside this checkout.' >&2; exit 2 ;; esac
host_session="$(mktemp -d "$state/session.XXXXXX")"
session="$host_session"
runner_home="$host_session/home"
mkdir -p "$runner_home/.codex" "$host_session/docker-client"
printf '{"auths":{}}\n' > "$host_session/docker-client/config.json"
docker_cmd=("$docker_bin" --config "$host_session/docker-client" --host "$endpoint")
profile="${RESEARCH_RUNNER_CODEX_PROFILE:-agsci-magpie}"
case "$profile" in ''|*[!a-zA-Z0-9_-]*) echo 'Invalid Codex profile name.' >&2; exit 2 ;; esac
auth_root="${RESEARCH_RUNNER_AUTH_ROOT:-$root/scratch/harness-audit/container-auth}"
config="$auth_root/.codex/$profile.config.toml"
if [ -f "$config" ]; then
  cp "$config" "$runner_home/.codex/$profile.config.toml"
  cp "$config" "$runner_home/.codex/config.toml"
fi
source_mount="type=bind,source=$root,target=$root"
native_mount=()
native_root=""
if [ "$native" = 1 ]; then
  checkout_hash="$(printf '%s' "$root" | git hash-object --stdin)"
  volume="${RESEARCH_RUNNER_VOLUME:-agsci-${checkout_hash:0:12}}"
  case "$volume" in ''|*[!a-zA-Z0-9_.-]*) echo 'Invalid runner volume name.' >&2; exit 2 ;; esac
  if ! "${docker_cmd[@]}" volume inspect "$volume" >/dev/null 2>&1; then
    "${docker_cmd[@]}" volume create --label agsci.owner=research-harness \
      --label "agsci.checkout=$root" "$volume" >/dev/null
  fi
  owner="$("${docker_cmd[@]}" volume inspect "$volume" --format '{{index .Labels "agsci.owner"}}')"
  checkout="$("${docker_cmd[@]}" volume inspect "$volume" --format '{{index .Labels "agsci.checkout"}}')"
  if [ "$owner" != research-harness ] || [ "$checkout" != "$root" ]; then
    echo 'Use a dedicated runner volume belonging to this checkout.' >&2
    exit 2
  fi
  native_root="$("${docker_cmd[@]}" volume inspect "$volume" --format '{{.Mountpoint}}')"
  # Change only the owned volume root, never recursively chown user data.
  "${docker_cmd[@]}" run --rm --network none --user 0:0 \
    --mount "type=volume,source=$volume,target=/state" --entrypoint /usr/local/bin/python "$image" -c \
    'import os,sys; uid,gid=map(int,sys.argv[1:]); owner=os.stat("/state",follow_symlinks=False).st_uid; assert owner in (0,uid), "native volume belongs to another UID"; os.chown("/state",uid,gid); os.chmod("/state",0o700)' \
    "$(id -u)" "$(id -g)"
  source_mount="$source_mount,readonly"
  native_mount=(--mount "type=volume,source=$volume,target=$native_root")
  helper="$root/src/docker/research-runner/native_state.py"
  session="$("${docker_cmd[@]}" run --rm --network none --user "$(id -u):$(id -g)" \
    --mount "$source_mount" "${native_mount[@]}" --entrypoint /usr/local/bin/python "$image" \
    "$helper" init "$native_root" "$runner_home" "$profile")"
  runner_home="$session/home"
  temp_root="$native_root/tmp"
  if [ "${1:-}" = --state-info ]; then
    exec "${docker_cmd[@]}" run --rm --network none --user "$(id -u):$(id -g)" \
      --mount "$source_mount" "${native_mount[@]}" --entrypoint /usr/local/bin/python "$image" \
      "$helper" info "$native_root" "$volume" "$root"
  fi
  if [ "${1:-}" = --export-workspace ]; then
    if [ "$#" -ne 3 ]; then echo '--export-workspace requires a relative workspace and a new host directory.' >&2; exit 2; fi
    destination="$3"
    if [ -e "$destination" ] || [ -L "$destination" ]; then echo 'Export destination must not exist.' >&2; exit 2; fi
    mkdir -p "$destination"
    destination="$(cd "$destination" && pwd -P)"
    "${docker_cmd[@]}" run --rm --network none --user "$(id -u):$(id -g)" \
      --mount "$source_mount" --mount "type=volume,source=$volume,target=$native_root,readonly" \
      --mount "type=bind,source=$destination,target=/export" --entrypoint /usr/local/bin/python "$image" \
      "$helper" export "$native_root" "$2" /export
    # Docker Desktop may ignore chmod through its shared filesystem. Apply the
    # snapshot permissions on the host, without following exported symlinks.
    find "$destination" -type f -exec chmod a-w {} +
    find "$destination" -type d -exec chmod a-w {} +
    printf 'Exported read-only snapshot: %s\n' "$destination"
    exit 0
  fi
else
  temp_root="${AGENTIC_TEMP_ROOT:-$root/scratch/tmp}"
  mkdir -p "$temp_root"
  temp_root="$(cd "$temp_root" && pwd -P)"
  case "$temp_root/" in "$root/"*) ;; *) echo 'AGENTIC_TEMP_ROOT must be inside this checkout.' >&2; exit 2 ;; esac
fi
# The socket is root:root inside Docker Desktop. Supplemental group 0 grants
# access without running the orchestrator as root. It is not sent to children.
socket_group="${RESEARCH_DOCKER_SOCKET_GID:-}"
if [ -z "$socket_group" ]; then
  if [ "$(uname -s)" = Darwin ]; then
    socket_group=0
  else
    socket_group="$(stat -c '%g' "$socket")"
  fi
fi
args=(run --rm --init --user "$(id -u):$(id -g)" --group-add "$socket_group"
  --entrypoint "$root/src/docker/research-runner/entrypoint.sh"
  --mount "$source_mount"
  --mount "type=bind,source=$socket,target=/var/run/docker.sock"
  --workdir "$root" --add-host host.docker.internal:host-gateway
  --env "HOME=$runner_home" --env "CODEX_HOME=$runner_home/.codex"
  --env USER=agent --env LOGNAME=agent
  --env "DOCKER_CONFIG=$session/docker-client" --env DOCKER_HOST=unix:///var/run/docker.sock
  --env "AGENTIC_TEMP_ROOT=$temp_root" --env "TMPDIR=$temp_root"
  --env "AGENTIC_TASK_PACKAGE_INDEX=${AGENTIC_TASK_PACKAGE_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
  --env "RESEARCH_RUNNER_CODEX_PROFILE=$profile"
  --env "RESEARCH_RUNNER_NATIVE_ROOT=$native_root"
  --env "RESEARCH_CPU_IMAGE=${RESEARCH_CPU_IMAGE:-unified-ai-scientist/research-cpu:2026.10}"
  --env HTTP_PROXY= --env HTTPS_PROXY= --env ALL_PROXY=
  --env http_proxy= --env https_proxy= --env all_proxy=)
if [ "$native" = 1 ]; then args+=("${native_mount[@]}"); fi
if [ -n "${RESEARCH_RUNNER_ENV_FILE:-}" ]; then
  if [ ! -f "$RESEARCH_RUNNER_ENV_FILE" ] || [ ! -r "$RESEARCH_RUNNER_ENV_FILE" ]; then
    echo 'RESEARCH_RUNNER_ENV_FILE must be a readable file.' >&2
    exit 2
  fi
  args+=(--env-file "$RESEARCH_RUNNER_ENV_FILE")
fi
# Managed worktrees refer to the original repository's Git metadata. Expose it
# at the same absolute path, read-only, for revision stamps and discovery.
git_common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)"
case "$git_common/" in
  "$root/"*) ;;
  *) args+=(--mount "type=bind,source=$git_common,target=$git_common,readonly") ;;
esac
printf 'Linux runner session: %s\n' "$session" >&2
if [ "$#" -eq 0 ]; then set -- --help; fi
exec "${docker_cmd[@]}" "${args[@]}" "$image" "$@"
