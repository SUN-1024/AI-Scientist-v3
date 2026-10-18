#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -s)" != Linux ]; then
  echo 'The research runner requires Linux.' >&2
  exit 2
fi
# Mapped host UIDs need not have an /etc/passwd entry in the image.
export USER="${USER:-agent}"
export LOGNAME="${LOGNAME:-$USER}"
if [ -n "${RESEARCH_RUNNER_NATIVE_ROOT:-}" ]; then
  python "$(dirname "${BASH_SOURCE[0]}")/native_state.py" validate "$RESEARCH_RUNNER_NATIVE_ROOT" "$@"
fi
case "${1:-}" in
  --preflight)
    shift
    exec python "$(dirname "${BASH_SOURCE[0]}")/preflight.py" "$@"
    ;;
  --exec)
    shift
    if [ "$#" -eq 0 ]; then
      echo '--exec requires a command.' >&2
      exit 2
    fi
    exec "$@"
    ;;
esac
exec python -m src.agsci.main "$@"
