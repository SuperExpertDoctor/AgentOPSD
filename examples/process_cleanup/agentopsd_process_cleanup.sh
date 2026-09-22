#!/usr/bin/env bash

agentopsd_cleanup_setup() {
    local python_bin="${1:?python executable is required}"
    local hook_dir repo_root launcher_path cleanup_program

    hook_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    repo_root="$(cd -- "${hook_dir}/../.." && pwd -P)"
    launcher_path="$(readlink -f -- "${BASH_SOURCE[1]:-$0}")"
    cleanup_program="${AGENTOPSD_CLEANUP_PROGRAM:-${hook_dir}/agentopsd_process_cleanup.py}"

    export AGENTOPSD_RUN_ID="agentopsd-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
    export AGENTOPSD_LAUNCHER_PID="$$"
    export AGENTOPSD_REPO_ROOT="$repo_root"
    export AGENTOPSD_LAUNCHER_PATH="$launcher_path"
    export AGENTOPSD_CLEANUP_PYTHON="$python_bin"
    export AGENTOPSD_CLEANUP_PROGRAM="$cleanup_program"

    "$python_bin" "$cleanup_program" preflight \
        --repo-root "$repo_root" \
        --launcher "$launcher_path" \
        --launcher-pid "$$" \
        --run-id "$AGENTOPSD_RUN_ID"

    trap '_agentopsd_cleanup_on_exit $?' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}

_agentopsd_cleanup_on_exit() {
    local exit_code="${1:-0}"
    trap - EXIT INT TERM
    if declare -F agentopsd_search_cleanup >/dev/null; then
        if ! agentopsd_search_cleanup; then
            echo "Retrieval cleanup could not be verified." >&2
            if [[ "$exit_code" -eq 0 ]]; then exit_code=1; fi
        fi
    fi
    "$AGENTOPSD_CLEANUP_PYTHON" "$AGENTOPSD_CLEANUP_PROGRAM" cleanup \
        --repo-root "$AGENTOPSD_REPO_ROOT" \
        --launcher "$AGENTOPSD_LAUNCHER_PATH" \
        --launcher-pid "$AGENTOPSD_LAUNCHER_PID" \
        --run-id "$AGENTOPSD_RUN_ID" || true
    exit "$exit_code"
}
