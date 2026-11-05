#!/usr/bin/env bash
# Apply this checkout's network settings to a command and its child processes.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
proxy_config="${project_root}/.env.proxy"

if [[ ! -f "$proxy_config" ]]; then
    printf 'Create %s with MUSCLEMIMIC_HTTP_PROXY set to your HTTP proxy URL.\n' "$proxy_config" >&2
    exit 1
fi

source "$proxy_config"
proxy_url="${MUSCLEMIMIC_HTTP_PROXY:-}"
case "$proxy_url" in
    http://*|https://*) ;;
    *)
        printf 'MUSCLEMIMIC_HTTP_PROXY must start with http:// or https://.\n' >&2
        exit 1
        ;;
esac

if (( $# == 0 )); then
    printf 'Usage: %s COMMAND [ARGUMENTS...]\n' "$0" >&2
    exit 2
fi

unset all_proxy ALL_PROXY
export http_proxy="$proxy_url" https_proxy="$proxy_url"
export HTTP_PROXY="$proxy_url" HTTPS_PROXY="$proxy_url"
exec "$@"
