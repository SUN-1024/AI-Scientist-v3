#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="${RESEARCH_RUNNER_IMAGE:-unified-ai-scientist/research-runner:2026.10}"
base_image="${RESEARCH_CPU_IMAGE:-unified-ai-scientist/research-cpu:2026.10}"
docker_bin="${DOCKER_BIN:-docker}"
if ! command -v "$docker_bin" >/dev/null 2>&1; then
  docker_bin=/Applications/Docker.app/Contents/Resources/bin/docker
fi
command -v "$docker_bin" >/dev/null 2>&1 || { echo 'Docker is required.' >&2; exit 2; }
command -v uv >/dev/null 2>&1 || { echo 'uv is required to export uv.lock.' >&2; exit 2; }
context="$(mktemp -d "${TMPDIR:-/tmp}/research-runner-build.XXXXXX")"
trap 'rm -rf "$context"' EXIT
mkdir -p "$context/docker-client"
printf '{"auths":{}}\n' > "$context/docker-client/config.json"
# An explicit endpoint allows a minimal client config without copying host
# credentials, credential helpers, proxy configuration, or global settings.
endpoint="${DOCKER_HOST:-$("$docker_bin" context inspect --format '{{.Endpoints.docker.Host}}')}"
uv export --project "$root" --frozen --all-groups --no-emit-project \
  --no-header --format requirements-txt --output-file "$context/requirements.lock.txt" >/dev/null
cp "$root/src/docker/research-runner/"{Dockerfile,entrypoint.sh,preflight.py,native_state.py} "$context/"
"$docker_bin" --config "$context/docker-client" --host "$endpoint" build -t "$image" \
  --build-arg "RESEARCH_CPU_IMAGE=$base_image" \
  --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg ALL_PROXY= \
  --build-arg http_proxy= --build-arg https_proxy= --build-arg all_proxy= \
  "$@" "$context"
"$docker_bin" --config "$context/docker-client" --host "$endpoint" image inspect \
  "$image" --format 'Built {{.Id}} ({{.Os}}/{{.Architecture}})'
