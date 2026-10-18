#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision=93cd9b06b85f848af1308c41e018991b33907c5e
image="${RESEARCH_CPU_IMAGE:-unified-ai-scientist/research-cpu:2026.10}"
docker_bin="${DOCKER_BIN:-docker}"
if ! command -v "$docker_bin" >/dev/null 2>&1; then
  docker_bin=/Applications/Docker.app/Contents/Resources/bin/docker
fi
command -v "$docker_bin" >/dev/null 2>&1 || { echo 'Docker is required.' >&2; exit 1; }
context="$(mktemp -d "${TMPDIR:-/tmp}/research-cpu-image.XXXXXX")"
trap 'rm -rf "$context"' EXIT
source_dir="${HEADLESS_SOURCE:-}"
if [ -z "$source_dir" ]; then
  source_dir="$context/source"
  git -c http.proxy="${RESEARCH_DOWNLOAD_PROXY:-http://127.0.0.1:7890}" \
    clone --filter=blob:none https://github.com/RobertTLange/headless-cli.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
mkdir "$context/headless"
git -C "$source_dir" archive "$revision" | tar -x -C "$context/headless"
cp "$root/src/docker/research-cpu/Dockerfile" "$context/Dockerfile"
# Docker Desktop can inject a host loopback proxy into build containers. Start
# direct for the mainland mirrors. Explicit caller build arguments override this.
"$docker_bin" build -t "$image" \
  --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= --build-arg ALL_PROXY= \
  --build-arg http_proxy= --build-arg https_proxy= --build-arg all_proxy= \
  "$@" "$context"
printf 'Built %s from Headless %s\n' "$image" "$revision"
