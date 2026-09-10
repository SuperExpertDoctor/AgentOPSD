import json
import os
import signal
from pathlib import Path

import pytest

from examples.process_cleanup import agentopsd_process_cleanup as cleanup


def process(
    pid,
    *,
    ppid=1,
    uid=1000,
    state="S",
    start_time=None,
    cwd="/repo",
    cmdline=("python",),
    run_id=None,
):
    return cleanup.ProcessInfo(
        pid=pid,
        ppid=ppid,
        uid=uid,
        state=state,
        start_time=start_time if start_time is not None else pid * 10,
        cwd=cwd,
        cmdline=tuple(cmdline),
        run_id=run_id,
    )


def test_selects_old_launcher_driver_descendants_and_workers_only():
    previous_launcher = cleanup.ProcessIdentity(pid=10, start_time=100)
    processes = [
        process(10, cmdline=("bash", "/repo/examples/agentopsd_trainer/run_alfworld_3b.sh")),
        process(11, ppid=10, cmdline=("python", "-m", "examples.data_preprocess.prepare")),
        process(20, cmdline=("python", "-m", "verl.trainer.main_opsd"), run_id="old-run"),
        process(21, ppid=20, cmdline=("env-child",)),
        process(30, cwd="/tmp/ray/session", cmdline=("ray::AlfworldWorker",), run_id="old-run"),
        process(31, cmdline=("ray::WorkerDict",)),
        process(40, cmdline=("raylet",), run_id="old-run"),
        process(41, uid=1001, cmdline=("python", "-m", "verl.trainer.main_opsd")),
        process(42, cwd="/other", cmdline=("python", "-m", "verl.trainer.main_opsd")),
        process(99, cmdline=("python", "/repo/examples/process_cleanup/agentopsd_process_cleanup.py")),
    ]

    selection = cleanup.select_preflight_targets(
        processes,
        repo_root="/repo",
        previous_run_id="old-run",
        previous_launcher=previous_launcher,
        current_uid=1000,
        ignored_pids={99},
    )

    assert [item.pid for item in selection.primary] == [10, 20]
    assert [item.pid for item in selection.remaining] == [11, 21, 30, 31]
    assert selection.old_run_ids == frozenset({"old-run"})


def test_rejects_reused_launcher_pid_and_substring_driver_matches():
    processes = [
        process(10, start_time=101, cmdline=("bash", "/repo/run.sh")),
        process(20, cmdline=("python", "tool.py", "verl.trainer.main_opsd")),
    ]

    selection = cleanup.select_preflight_targets(
        processes,
        repo_root="/repo",
        previous_run_id=None,
        previous_launcher=cleanup.ProcessIdentity(pid=10, start_time=100),
        current_uid=1000,
        ignored_pids=set(),
    )

    assert selection.primary == ()
    assert selection.remaining == ()


def test_parse_proc_stat_handles_spaces_in_process_name():
    tail_fields = ["S"] + ["0"] * 18 + ["987654"]
    stat_text = "123 (ray worker name) " + " ".join(tail_fields)

    assert cleanup._parse_start_time(stat_text) == 987654


def test_process_inspection_permission_error_is_not_treated_as_exit(monkeypatch, tmp_path):
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    monkeypatch.setattr(cleanup, "_read_status", lambda _path: (_ for _ in ()).throw(PermissionError("denied")))

    with pytest.raises(cleanup.ProcessInspectionError, match="pid 123"):
        cleanup.read_process(123, tmp_path)
