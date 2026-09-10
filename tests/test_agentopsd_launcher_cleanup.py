import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "examples/process_cleanup/agentopsd_process_cleanup.sh"
LAUNCHERS = sorted((REPO_ROOT / "examples/agentopsd_trainer").glob("*.sh"))


def test_exit_trap_runs_cleanup_and_preserves_status(tmp_path):
    fake_cli = tmp_path / "fake_cleanup.py"
    call_log = tmp_path / "calls.txt"
    fake_cli.write_text(
        "import os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(sys.argv[1] + '\\n')\n"
    )
    env = {
        **os.environ,
        "AGENTOPSD_CLEANUP_PROGRAM": str(fake_cli),
        "CALL_LOG": str(call_log),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; agentopsd_cleanup_setup "$2"; exit 17',
            "bash",
            str(HOOK),
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )

    assert result.returncode == 17
    assert call_log.read_text().splitlines() == ["preflight", "cleanup"]


def test_all_agentopsd_launchers_source_and_initialize_shared_hook():
    assert len(LAUNCHERS) == 7
    for launcher in LAUNCHERS:
        source = launcher.read_text()
        assert source.startswith("#!/usr/bin/env bash\n")
        assert "set -euo pipefail" in source
        assert "../process_cleanup/agentopsd_process_cleanup.sh" in source
        assert "agentopsd_cleanup_setup" in source
        assert "verl.trainer.agentopsd_cleanup preflight" not in source
        assert "flock" not in source
