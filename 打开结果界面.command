#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./scripts/local-agsci inspect start --no-build
