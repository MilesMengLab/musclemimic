#!/usr/bin/env bash
# Launch from this checkout even after its directory is renamed.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
entry="${1:-}"
case "$entry" in
    train|play|preview|export|fit) shift ;;
    *) printf 'Usage: %s {train|play|preview|export|fit} [arguments...]\n' "$0" >&2; exit 2 ;;
esac
cd "$project_root"
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    printf 'Create the environment first: uv sync --extra cuda\n' >&2
    exit 1
fi
if [[ "$entry" == "train" ]]; then
    module="hum_exo.train"
else
    module="scripts.hum_exo.$entry"
fi
exec "$project_root/.venv/bin/python" -m "$module" "$@"
