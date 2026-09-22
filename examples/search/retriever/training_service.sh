#!/usr/bin/env bash

agentopsd_search_setup() {
    local argument
    SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8081/retrieve}"
    for argument in "$@"; do
        case "$argument" in
            env.search.search_url=*) SEARCH_URL="${argument#env.search.search_url=}" ;;
        esac
    done
    export SEARCH_URL
    AGENTOPSD_RETRIEVAL_HELPER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/startup.py"
    AGENTOPSD_RETRIEVAL_STATE="$(mktemp -d "/tmp/agentopsd-retrieval-${UID}-$$-XXXXXXXX")/state.json"
    export AGENTOPSD_RETRIEVAL_HELPER AGENTOPSD_RETRIEVAL_STATE
    "${PYTHON_BIN:-python3}" "$AGENTOPSD_RETRIEVAL_HELPER" start \
        --owner-pid "$$" --state-file "$AGENTOPSD_RETRIEVAL_STATE" \
        --url "$SEARCH_URL" --gpu "${RETRIEVAL_GPU_ID:-6}" \
        --timeout "${RETRIEVAL_STARTUP_TIMEOUT:-1200}"
}

agentopsd_search_cleanup() {
    if [[ -n "${AGENTOPSD_RETRIEVAL_STATE:-}" ]]; then
        "${PYTHON_BIN:-python3}" "$AGENTOPSD_RETRIEVAL_HELPER" stop \
            --state-file "$AGENTOPSD_RETRIEVAL_STATE" || return "$?"
        rmdir -- "${AGENTOPSD_RETRIEVAL_STATE%/*}"
    fi
}
