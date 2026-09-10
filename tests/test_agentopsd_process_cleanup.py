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


def test_terminate_rechecks_start_time_before_sigkill(monkeypatch):
    original = process(20, start_time=200, cmdline=("python", "-m", "verl.trainer.main_opsd"))
    reused = process(20, start_time=201, cmdline=original.cmdline)
    signals = []
    reads = iter([original, reused])

    monkeypatch.setattr(cleanup.time, "sleep", lambda _seconds: None)
    result = cleanup.terminate_processes(
        [original],
        reader=lambda _pid: next(reads, reused),
        send_signal=lambda pid, sig: signals.append((pid, sig)),
        grace_period=0,
    )

    assert signals == [(20, signal.SIGTERM)]
    assert result.survivors == ()


def test_terminate_escalates_and_reports_a_survivor(monkeypatch):
    target = process(20, start_time=200)
    signals = []
    monkeypatch.setattr(cleanup.time, "sleep", lambda _seconds: None)

    result = cleanup.terminate_processes(
        [target],
        reader=lambda _pid: target,
        send_signal=lambda pid, sig: signals.append((pid, sig)),
        grace_period=0,
    )

    assert signals == [(20, signal.SIGTERM), (20, signal.SIGKILL)]
    assert result.survivors == (20,)


def test_state_namespace_is_per_user_and_repository(tmp_path):
    first = cleanup.state_paths("/repo/a", runtime_root=tmp_path, uid=1000)
    second = cleanup.state_paths("/repo/b", runtime_root=tmp_path, uid=1000)
    other_user = cleanup.state_paths("/repo/a", runtime_root=tmp_path, uid=1001)

    assert first != second
    assert first != other_user
    assert first.state.parent == first.lock.parent


def test_preflight_fails_when_a_newer_claim_replaces_it(monkeypatch, tmp_path):
    launcher = process(50, cmdline=("bash", "/repo/run.sh"))
    monkeypatch.setattr(cleanup.os, "getuid", lambda: 1000)
    calls = 0

    def replace_during_termination(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            cleanup.write_state(
                cleanup.RunState("newer", 60, 600, "/repo/new.sh", "/repo", "2026-09-10T00:00:01Z"),
                runtime_root=tmp_path,
            )
        return cleanup.TerminationResult((), (), ())

    monkeypatch.setattr(cleanup, "terminate_processes", replace_during_termination)

    with pytest.raises(cleanup.CleanupError, match="replaced by a newer launcher"):
        cleanup.preflight(
            repo_root="/repo",
            launcher_path="/repo/run.sh",
            launcher_pid=50,
            run_id="current",
            runtime_root=tmp_path,
            process_provider=lambda: [launcher],
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        json.dumps({"run_id": "x"}).encode(),
        json.dumps(
            {
                "run_id": "x",
                "launcher_pid": -1,
                "launcher_start_time": 10,
                "launcher_path": "/repo/run.sh",
                "repo_root": "/repo",
                "created_at": "2026-09-10T00:00:00Z",
            }
        ).encode(),
        json.dumps(
            {
                "run_id": "x",
                "launcher_pid": 10,
                "launcher_start_time": "wrong-type",
                "launcher_path": "/repo/run.sh",
                "repo_root": "/repo",
                "created_at": "2026-09-10T00:00:00Z",
            }
        ).encode(),
        json.dumps(
            {
                "run_id": "x",
                "launcher_pid": 10,
                "launcher_start_time": 100,
                "launcher_path": "/repo/run.sh",
                "repo_root": "/other",
                "created_at": "2026-09-10T00:00:00Z",
            }
        ).encode(),
    ],
)
def test_invalid_state_is_rejected_without_rewrite(tmp_path, payload):
    paths = cleanup.state_paths("/repo", runtime_root=tmp_path, uid=os.getuid())
    paths.directory.mkdir(parents=True)
    paths.state.write_bytes(payload)

    with pytest.raises(cleanup.CleanupError, match="state"):
        cleanup.load_state("/repo", runtime_root=tmp_path)

    assert paths.state.read_bytes() == payload
