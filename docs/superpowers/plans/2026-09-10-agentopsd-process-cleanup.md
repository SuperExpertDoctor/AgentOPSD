# AgentOPSD Residual Process Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repository-scoped cleanup mechanism under `examples` that makes every AgentOPSD launcher terminate an older launcher, driver, environment process, and training worker before starting a replacement run.

**Architecture:** A standard-library Python CLI owns `/proc` inspection, run-state claims, PID identity validation, and bounded TERM/KILL escalation. A sourced shell hook assigns each invocation a run id and invokes the CLI before launch and from an exit trap; `main_opsd` retains graceful in-process cleanup for resources that still have live Python/Ray handles.

**Tech Stack:** Python 3.10+, Linux `/proc`, `fcntl.flock`, POSIX signals, Bash, pytest, Ray lifecycle APIs already used by the repository.

## Global Constraints

- Eligible processes must have the current effective UID and belong to this resolved repository or to a validated run-id/launcher relationship.
- A newer launcher replaces an older launcher from this checkout; it does not merely refuse to start.
- Never call `ray stop`, broad `pkill`, or process-group termination.
- Never terminate `raylet`, `gcs_server`, `plasma_store`, Ray dashboard processes, or generic `default_worker.py` infrastructure.
- Revalidate UID, `/proc` start time, command line, working directory, and run id before every destructive signal.
- The preflight CLI must use only the Python standard library and must not import Ray, Torch, Hydra, or the training package.
- All seven current `examples/agentopsd_trainer/*.sh` launchers must use the shared hook and retain their existing training arguments and interpreter choices.
- Existing uncommitted edits are user-owned. Inspect them before each task, preserve them, and never stage unrelated hunks.
- Verification in this plan is CPU-only. Do not start training, inference, Ray clusters, environment simulations, or GPU processes.

---

## File Map

- Create `examples/process_cleanup/__init__.py`: marks the reusable cleanup package.
- Create `examples/process_cleanup/agentopsd_process_cleanup.py`: process inventory, target selection, run-state claims, signal escalation, and CLI.
- Create `examples/process_cleanup/agentopsd_process_cleanup.sh`: shared launcher setup and exit trap.
- Create `tests/test_agentopsd_process_cleanup.py`: standard-library cleaner unit tests.
- Create `tests/test_agentopsd_launcher_cleanup.py`: shell-hook and launcher integration tests.
- Modify `verl/trainer/agentopsd_cleanup.py`: retain only graceful cleanup of owned environment/Ray resources and the shared run-id constant.
- Modify `tests/test_agentopsd_cleanup.py`: keep runtime cleanup tests and remove the superseded “refuse live driver” behavior.
- Modify `verl/trainer/main_opsd.py`: propagate the run id and run graceful cleanup on every exit path.
- Modify `tests/test_main_opsd_device_setup.py`: verify run-id propagation plus runner/Ray teardown without starting Ray.
- Modify `agent_system/environments/env_package/alfworld/envs.py`: idempotently close actor-held environments and kill all owned actors.
- Modify `agent_system/environments/env_package/webshop/envs.py`: propagate run id to actors and make partial/duplicate close safe.
- Modify `tests/test_agentopsd_local_assets.py`: verify environment cleanup wiring and all launcher integrations.
- Modify every file in `examples/agentopsd_trainer/*.sh`: source and initialize the common hook.

---

### Task 1: Process Inventory and Target Selection

**Files:**
- Create: `examples/process_cleanup/__init__.py`
- Create: `examples/process_cleanup/agentopsd_process_cleanup.py`
- Create: `tests/test_agentopsd_process_cleanup.py`

**Interfaces:**
- Produces: `ProcessInfo`, `ProcessIdentity`, `ProcessInspectionError`, `Selection`, `iter_processes()`, `read_process()`, and `select_preflight_targets()`.
- Consumes: Linux `/proc/<pid>/{status,stat,cmdline,cwd,environ}` only.

- [ ] **Step 1: Write failing selection tests with synthetic process records**

Create the package marker and start `tests/test_agentopsd_process_cleanup.py` with concrete fixtures that represent an old launcher tree, a detached Ray actor carrying the old run id, a legacy worker, shared Ray infrastructure, and unrelated processes:

```python
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
```

- [ ] **Step 2: Run the tests and verify the missing module/API is the failure**

Run:

```bash
pytest -q tests/test_agentopsd_process_cleanup.py
```

Expected: collection fails because `examples.process_cleanup` or the declared dataclasses/functions do not exist.

- [ ] **Step 3: Implement process records, `/proc` reading, and conservative selection**

Implement these exact public records and selection contract in `examples/process_cleanup/agentopsd_process_cleanup.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

RUN_ID_ENV = "AGENTOPSD_RUN_ID"
DRIVER_MODULE = "verl.trainer.main_opsd"
INFRASTRUCTURE_MARKERS = (
    "raylet",
    "gcs_server",
    "plasma_store",
    "dashboard",
    "ray_dashboard",
    "default_worker.py",
)
LEGACY_WORKER_MARKERS = (
    "ray::OPSDTaskRunner",
    "ray::WorkerDict",
    "ray::AlfworldWorker",
    "ray::WebshopWorker",
)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_time: int


class ProcessInspectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    uid: int | None
    state: str
    start_time: int
    cwd: str
    cmdline: tuple[str, ...]
    run_id: str | None

    @property
    def identity(self) -> ProcessIdentity:
        return ProcessIdentity(self.pid, self.start_time)

    @property
    def command(self) -> str:
        return " ".join(self.cmdline)


@dataclass(frozen=True)
class Selection:
    primary: tuple[ProcessInfo, ...]
    remaining: tuple[ProcessInfo, ...]
    old_run_ids: frozenset[str]


def _has_python_module(process: ProcessInfo, module: str) -> bool:
    return any(
        arg == "-m" and index + 1 < len(process.cmdline) and process.cmdline[index + 1] == module
        for index, arg in enumerate(process.cmdline)
    )


def _is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((os.path.realpath(path), root)) == root
    except ValueError:
        return False


def _belongs_to_repo(process: ProcessInfo, repo_root: str) -> bool:
    root = os.path.realpath(repo_root)
    if _is_under(process.cwd, root):
        return True
    return any(os.path.isabs(arg) and _is_under(arg, root) for arg in process.cmdline)


def _is_infrastructure(process: ProcessInfo) -> bool:
    command = process.command.lower()
    return any(marker.lower() in command for marker in INFRASTRUCTURE_MARKERS)


def _is_legacy_worker(process: ProcessInfo, repo_root: str) -> bool:
    return _belongs_to_repo(process, repo_root) and any(
        marker in process.command for marker in LEGACY_WORKER_MARKERS
    )


def _descendant_pids(processes: Sequence[ProcessInfo], roots: set[int]) -> set[int]:
    descendants = set()
    frontier = set(roots)
    while frontier:
        children = {item.pid for item in processes if item.ppid in frontier}
        children -= descendants
        descendants.update(children)
        frontier = children
    return descendants


def select_preflight_targets(
    processes: Sequence[ProcessInfo],
    *,
    repo_root: str,
    previous_run_id: str | None,
    previous_launcher: ProcessIdentity | None,
    current_uid: int,
    ignored_pids: set[int],
) -> Selection:
    eligible = [
        item
        for item in processes
        if item.uid == current_uid
        and item.state != "Z"
        and item.pid not in ignored_pids
        and not _is_infrastructure(item)
    ]
    by_pid = {item.pid: item for item in eligible}
    old_launcher = None
    if previous_launcher is not None:
        candidate = by_pid.get(previous_launcher.pid)
        if candidate is not None and candidate.start_time == previous_launcher.start_time:
            old_launcher = candidate

    drivers = [
        item
        for item in eligible
        if _belongs_to_repo(item, repo_root) and _has_python_module(item, DRIVER_MODULE)
    ]
    primary = ([old_launcher] if old_launcher is not None else []) + drivers
    primary_by_pid = {item.pid: item for item in primary}
    old_run_ids = {item.run_id for item in drivers if item.run_id}
    if previous_run_id:
        old_run_ids.add(previous_run_id)

    descendants = _descendant_pids(eligible, set(primary_by_pid))
    remaining = [
        item
        for item in eligible
        if item.pid not in primary_by_pid
        and (
            item.pid in descendants
            or (item.run_id is not None and item.run_id in old_run_ids)
            or _is_legacy_worker(item, repo_root)
        )
    ]
    return Selection(
        primary=tuple(sorted(primary_by_pid.values(), key=lambda item: item.pid)),
        remaining=tuple(sorted(remaining, key=lambda item: item.pid)),
        old_run_ids=frozenset(old_run_ids),
    )
```

Add the `/proc` readers in the same module; do not use `ps` output or shell parsing:

```python
def _read_status(path: Path) -> tuple[int | None, int, str]:
    uid = None
    ppid = 0
    state = "?"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        fields = value.strip().split()
        if key == "Uid" and fields:
            uid = int(fields[0])
        elif key == "PPid" and fields:
            ppid = int(fields[0])
        elif key == "State" and fields:
            state = fields[0][0]
    return uid, ppid, state


def _parse_start_time(stat_text: str) -> int:
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("missing process name terminator")
    tail_fields = stat_text[closing_parenthesis + 1 :].split()
    if len(tail_fields) <= 19:
        raise ValueError("missing process start time")
    return int(tail_fields[19])


def _read_environment_value(path: Path, key: str) -> str | None:
    prefix = f"{key}=".encode()
    entries = path.read_bytes().split(b"\0")
    for item in entries:
        if item.startswith(prefix):
            return item[len(prefix) :].decode(errors="replace")
    return None


def read_process(pid: int, proc_root: str | os.PathLike[str] = "/proc") -> ProcessInfo | None:
    directory = Path(proc_root) / str(pid)
    try:
        uid, ppid, state = _read_status(directory / "status")
        start_time = _parse_start_time((directory / "stat").read_text(encoding="utf-8", errors="replace"))
        cmdline = tuple(
            item.decode(errors="replace")
            for item in (directory / "cmdline").read_bytes().split(b"\0")
            if item
        )
        cwd = os.readlink(directory / "cwd")
        run_id = _read_environment_value(directory / "environ", RUN_ID_ENV)
    except FileNotFoundError:
        return None
    except (PermissionError, OSError, ValueError) as exc:
        raise ProcessInspectionError(f"cannot inspect pid {pid}: {exc}") from exc
    if not cmdline:
        return None
    return ProcessInfo(pid, ppid, uid, state, start_time, cwd, cmdline, run_id)


def iter_processes(proc_root: str | os.PathLike[str] = "/proc") -> Iterable[ProcessInfo]:
    root = Path(proc_root)
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise ProcessInspectionError(f"cannot inspect process table: {exc}") from exc
    for entry in entries:
        if entry.name.isdigit():
            item = read_process(int(entry.name), root)
            if item is not None:
                yield item
```

- [ ] **Step 4: Run selection tests and the import compiler check**

Run:

```bash
pytest -q tests/test_agentopsd_process_cleanup.py
python -m py_compile examples/process_cleanup/agentopsd_process_cleanup.py
```

Expected: both commands exit 0; pytest reports 4 passed.

- [ ] **Step 5: Checkpoint the isolated new files**

Run:

```bash
git diff --check -- examples/process_cleanup tests/test_agentopsd_process_cleanup.py
git add -- examples/process_cleanup/__init__.py examples/process_cleanup/agentopsd_process_cleanup.py tests/test_agentopsd_process_cleanup.py
git diff --cached --check
git commit -m "feat: identify stale AgentOPSD processes"
```

Expected: the staged diff contains only the three new Task 1 files. If any path existed before execution or contains user-owned edits, do not commit it; leave the verified changes unstaged and report that exception at the checkpoint.

---

### Task 2: PID-Safe Termination and Run-State Claim Protocol

**Files:**
- Modify: `examples/process_cleanup/agentopsd_process_cleanup.py`
- Modify: `tests/test_agentopsd_process_cleanup.py`

**Interfaces:**
- Consumes: Task 1 `ProcessInfo`, `ProcessIdentity`, `Selection`, `read_process()`, and `select_preflight_targets()`.
- Produces: `RunState`, `CleanupError`, `state_paths()`, `load_state()`, `preflight()`, `cleanup_run()`, and CLI actions `preflight`/`cleanup`.

- [ ] **Step 1: Add failing tests for PID reuse, escalation, state isolation, and newest-claim-wins**

Append tests using injected providers and signal functions rather than signaling real processes:

```python
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
```

The parameterized state test covers malformed JSON, missing fields, a nonpositive PID, a noninteger start time, and a mismatched repository root without changing the file.

- [ ] **Step 2: Run the focused tests and verify the new APIs are missing**

Run:

```bash
pytest -q tests/test_agentopsd_process_cleanup.py
```

Expected: Task 1 tests pass and the new tests fail on missing `terminate_processes`, `RunState`, or `state_paths` behavior.

- [ ] **Step 3: Implement identity checks and bounded TERM/KILL escalation**

Add these public result/error types and preserve the exact identity comparison:

```python
import signal
import time


class CleanupError(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminationResult:
    terminated: tuple[int, ...]
    killed: tuple[int, ...]
    survivors: tuple[int, ...]


def same_process(current: ProcessInfo | None, expected: ProcessInfo) -> bool:
    return current is not None and current.state != "Z" and (
        current.pid,
        current.uid,
        current.start_time,
        current.cwd,
        current.cmdline,
        current.run_id,
    ) == (
        expected.pid,
        expected.uid,
        expected.start_time,
        expected.cwd,
        expected.cmdline,
        expected.run_id,
    )


def terminate_processes(
    processes,
    *,
    reader=read_process,
    send_signal=os.kill,
    grace_period=2.0,
):
    captured = {item.pid: item for item in processes}
    terminated = []
    for item in captured.values():
        if same_process(reader(item.pid), item):
            try:
                send_signal(item.pid, signal.SIGTERM)
                terminated.append(item.pid)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + max(0.0, grace_period)
    survivors = [item for item in captured.values() if same_process(reader(item.pid), item)]
    while survivors and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        survivors = [item for item in survivors if same_process(reader(item.pid), item)]

    killed = []
    for item in survivors:
        if same_process(reader(item.pid), item):
            try:
                send_signal(item.pid, signal.SIGKILL)
                killed.append(item.pid)
            except ProcessLookupError:
                pass
    kill_deadline = time.monotonic() + max(0.0, grace_period)
    final = [item for item in survivors if same_process(reader(item.pid), item)]
    while final and time.monotonic() < kill_deadline:
        time.sleep(min(0.05, max(0.0, kill_deadline - time.monotonic())))
        final = [item for item in final if same_process(reader(item.pid), item)]
    final_survivors = tuple(item.pid for item in final)
    return TerminationResult(tuple(terminated), tuple(killed), final_survivors)
```

Catch `PermissionError`/`OSError` as cleanup failures instead of silently treating those processes as gone. Signal `Selection.primary` first, wait, then process a refreshed set of `Selection.remaining`; never signal by PGID.

- [ ] **Step 4: Implement locked, atomic state and two-phase ownership validation**

Add immutable records and path derivation:

```python
import contextlib
import fcntl
import hashlib
import json
import tempfile
from datetime import datetime, timezone


@dataclass(frozen=True)
class StatePaths:
    directory: Path
    state: Path
    lock: Path


@dataclass(frozen=True)
class RunState:
    run_id: str
    launcher_pid: int
    launcher_start_time: int
    launcher_path: str
    repo_root: str
    created_at: str


def state_paths(repo_root, *, runtime_root=None, uid=None):
    resolved = os.path.realpath(repo_root)
    owner = os.getuid() if uid is None else uid
    base = Path(runtime_root or os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or "/tmp")
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    directory = base / f"agentopsd-{owner}" / digest
    return StatePaths(directory, directory / "run.json", directory / "run.lock")
```

Implement locked state access and atomic replacement with these concrete helpers:

```python
@contextlib.contextmanager
def locked_state(repo_root, *, runtime_root=None):
    paths = state_paths(repo_root, runtime_root=runtime_root)
    paths.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(paths.directory, 0o700)
    descriptor = os.open(paths.lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield paths
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_state_path(path, repo_root):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = RunState(
            run_id=payload["run_id"],
            launcher_pid=payload["launcher_pid"],
            launcher_start_time=payload["launcher_start_time"],
            launcher_path=payload["launcher_path"],
            repo_root=payload["repo_root"],
            created_at=payload["created_at"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        raise CleanupError(f"invalid AgentOPSD run state: {exc}") from exc
    if not state.run_id or not isinstance(state.run_id, str):
        raise CleanupError("invalid AgentOPSD run state: run_id")
    if not isinstance(state.launcher_pid, int) or state.launcher_pid <= 0:
        raise CleanupError("invalid AgentOPSD run state: launcher_pid")
    if not isinstance(state.launcher_start_time, int) or state.launcher_start_time <= 0:
        raise CleanupError("invalid AgentOPSD run state: launcher_start_time")
    if not isinstance(state.launcher_path, str) or not state.launcher_path:
        raise CleanupError("invalid AgentOPSD run state: launcher_path")
    if not isinstance(state.repo_root, str) or not state.repo_root:
        raise CleanupError("invalid AgentOPSD run state: repo_root")
    if not isinstance(state.created_at, str) or not state.created_at:
        raise CleanupError("invalid AgentOPSD run state: created_at")
    if os.path.realpath(state.repo_root) != os.path.realpath(repo_root):
        raise CleanupError("run state belongs to a different repository")
    return state


def _write_state_path(path, state):
    payload = {
        "run_id": state.run_id,
        "launcher_pid": state.launcher_pid,
        "launcher_start_time": state.launcher_start_time,
        "launcher_path": state.launcher_path,
        "repo_root": state.repo_root,
        "created_at": state.created_at,
    }
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".run-",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o600)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_state(repo_root, *, runtime_root=None):
    with locked_state(repo_root, runtime_root=runtime_root) as paths:
        return _load_state_path(paths.state, repo_root)


def write_state(state, *, runtime_root=None):
    with locked_state(state.repo_root, runtime_root=runtime_root) as paths:
        _write_state_path(paths.state, state)
```

The validation errors must include the word `state` so callers get one stable, credential-free failure category.

Implement `preflight()` with this ordering:

1. Under `locked_state`, load the previous state, scan one snapshot, validate the current launcher record, atomically write this launcher's new `RunState`, and release the lock.
2. Select old launcher/driver/worker targets from the captured snapshot, signal primary targets, refresh survivors, then signal remaining targets.
3. Under `locked_state`, require the state run id and launcher identity still match this invocation; take a second snapshot while ownership is confirmed, then release.
4. Select and terminate old-run processes that appeared during shutdown.
5. Under `locked_state`, repeat the ownership check before returning success.
6. On any failure, remove state only if it still matches this invocation, then raise `CleanupError`.

Implement `cleanup_run()` to select only processes with the supplied run id, exclude the current launcher/cleaner and infrastructure, perform identity-safe termination, and remove state under lock only when its run id and launcher identity match.

- [ ] **Step 5: Add and implement the standard-library CLI**

Expose exactly these commands:

```text
python examples/process_cleanup/agentopsd_process_cleanup.py preflight \
  --repo-root PATH --launcher PATH --launcher-pid PID --run-id ID

python examples/process_cleanup/agentopsd_process_cleanup.py cleanup \
  --repo-root PATH --launcher PATH --launcher-pid PID --run-id ID
```

`main(argv=None) -> int` returns `0` after printing only action, run id, counts, and PIDs. It catches `CleanupError`, `ProcessInspectionError`, `PermissionError`, and signal-related `OSError`, prints a credential-free message to stderr, and returns `2`. It must never print a full command line or environment.

- [ ] **Step 6: Run the complete cleaner test module**

Run:

```bash
pytest -q tests/test_agentopsd_process_cleanup.py
python examples/process_cleanup/agentopsd_process_cleanup.py --help
```

Expected: all tests pass and help lists `preflight` and `cleanup`.

- [ ] **Step 7: Checkpoint Task 2**

Run:

```bash
git diff --check -- examples/process_cleanup tests/test_agentopsd_process_cleanup.py
git add -- examples/process_cleanup/agentopsd_process_cleanup.py tests/test_agentopsd_process_cleanup.py
git diff --cached --check
git commit -m "feat: replace stale AgentOPSD runs safely"
```

Expected: the staged diff contains only Task 2 changes. Preserve any pre-existing user hunks by leaving them unstaged.

---

### Task 3: Shared Shell Hook and Seven Launcher Integrations

**Files:**
- Create: `examples/process_cleanup/agentopsd_process_cleanup.sh`
- Create: `tests/test_agentopsd_launcher_cleanup.py`
- Modify: `examples/agentopsd_trainer/run_alfworld_3b.sh`
- Modify: `examples/agentopsd_trainer/run_alfworld_7b.sh`
- Modify: `examples/agentopsd_trainer/run_alfworld_7b_gpu1.sh`
- Modify: `examples/agentopsd_trainer/run_search_3b.sh`
- Modify: `examples/agentopsd_trainer/run_search_7b.sh`
- Modify: `examples/agentopsd_trainer/run_webshop_3b.sh`
- Modify: `examples/agentopsd_trainer/run_webshop_7b.sh`
- Modify: `tests/test_agentopsd_local_assets.py`

**Interfaces:**
- Consumes: Task 2 CLI `preflight` and `cleanup` commands.
- Produces: Bash function `agentopsd_cleanup_setup PYTHON_BIN` and exported `AGENTOPSD_RUN_ID`/`AGENTOPSD_LAUNCHER_PID`.

- [ ] **Step 1: Write failing shell-hook and static launcher tests**

Create `tests/test_agentopsd_launcher_cleanup.py` with a fake standard-library CLI that records actions, then assert setup calls preflight and the EXIT trap calls cleanup while preserving status 17:

```python
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
```

Update `test_agentopsd_scripts_use_local_models_and_datasets()` so the old `run_alfworld_3b.sh`-specific inline `flock`, `preflight`, and `trap cleanup` assertions are replaced with shared-hook assertions for every launcher.

- [ ] **Step 2: Run the shell tests and confirm the hook/integration is absent**

Run:

```bash
pytest -q tests/test_agentopsd_launcher_cleanup.py tests/test_agentopsd_local_assets.py
```

Expected: failures identify the missing hook and the six launchers not yet integrated.

- [ ] **Step 3: Create the common Bash hook**

Create `examples/process_cleanup/agentopsd_process_cleanup.sh` with the following behavior and no training-specific arguments:

```bash
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
    "$AGENTOPSD_CLEANUP_PYTHON" "$AGENTOPSD_CLEANUP_PROGRAM" cleanup \
        --repo-root "$AGENTOPSD_REPO_ROOT" \
        --launcher "$AGENTOPSD_LAUNCHER_PATH" \
        --launcher-pid "$AGENTOPSD_LAUNCHER_PID" \
        --run-id "$AGENTOPSD_RUN_ID" || true
    exit "$exit_code"
}
```

Keep the override `AGENTOPSD_CLEANUP_PROGRAM` solely to permit an isolated shell test; production launchers do not set it.

- [ ] **Step 4: Integrate all launchers without altering training options**

For each of the seven launchers, ensure the first block is:

```bash
#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
source "${SCRIPT_DIR}/../process_cleanup/agentopsd_process_cleanup.sh"
agentopsd_cleanup_setup "$PYTHON_BIN"
```

Preserve the existing Conda default in `run_alfworld_3b.sh` by using its current value instead of `python3`. Preserve the existing `PYTHON_BIN` declaration in `run_alfworld_7b_gpu1.sh`. Replace each training/data-preparation `python3` invocation with `"$PYTHON_BIN"` so preflight and training use the same interpreter. Remove only the duplicated inline cleanup/lock block currently present in `run_alfworld_3b.sh`; do not change GPU, model, batch, logger, or Hydra arguments.

- [ ] **Step 5: Run shell syntax and integration tests**

Run:

```bash
bash -n examples/process_cleanup/agentopsd_process_cleanup.sh examples/agentopsd_trainer/*.sh
pytest -q tests/test_agentopsd_launcher_cleanup.py tests/test_agentopsd_local_assets.py
```

Expected: Bash exits 0 and both pytest modules pass.

- [ ] **Step 6: Checkpoint Task 3 without capturing unrelated script edits**

Run:

```bash
git diff --check -- examples/process_cleanup/agentopsd_process_cleanup.sh examples/agentopsd_trainer tests/test_agentopsd_launcher_cleanup.py tests/test_agentopsd_local_assets.py
git status --short
```

Stage the new hook, the new shell test, the six launchers that were clean at plan time, and only newly authored hunks from any already-dirty file. Inspect `git diff --cached` before committing:

```bash
git commit -m "feat: clean stale processes from AgentOPSD launchers"
```

Expected: no pre-existing device, asset, or training-tuning change is included. If clean noninteractive staging cannot isolate the new hunks, leave that file uncommitted and report it rather than staging user-owned changes.

---

### Task 4: Graceful Runtime and Environment Cleanup

**Files:**
- Modify: `verl/trainer/agentopsd_cleanup.py`
- Modify: `tests/test_agentopsd_cleanup.py`
- Modify: `verl/trainer/main_opsd.py`
- Modify: `tests/test_main_opsd_device_setup.py`
- Modify: `agent_system/environments/env_package/alfworld/envs.py`
- Modify: `agent_system/environments/env_package/webshop/envs.py`

**Interfaces:**
- Consumes: exported shell environment variable `AGENTOPSD_RUN_ID`.
- Produces: `cleanup_runtime_resources(trainer=None, envs=(), ray_module=None) -> tuple[str, ...]` and idempotent environment `close()` methods.

- [ ] **Step 1: Replace obsolete preflight expectations with runtime cleanup tests**

Keep `test_cleanup_runtime_resources_closes_envs_kills_unique_workers_and_placement_groups`. Remove tests that import `ProcessInfo` or `preflight_cleanup` from `verl.trainer.agentopsd_cleanup`, since that responsibility now lives under `examples`.

Add a WebShop close test that constructs the wrapper without starting Ray:

```python
def test_webshop_close_is_idempotent_and_kills_workers(monkeypatch):
    from agent_system.environments.env_package.webshop import envs as webshop_envs

    events = []

    class RemoteClose:
        def __init__(self, name):
            self.name = name

        def remote(self):
            events.append(("close", self.name))
            return self.name

    class Worker:
        def __init__(self, name):
            self.name = name
            self.close = RemoteClose(name)

    wrapper = object.__new__(webshop_envs.WebshopMultiProcessEnv)
    wrapper._workers = [Worker("a"), Worker("b")]
    wrapper._closed = False
    monkeypatch.setattr(webshop_envs.ray, "get", lambda refs, timeout=None: events.append(("get", tuple(refs))))
    monkeypatch.setattr(webshop_envs.ray, "kill", lambda worker, no_restart=True: events.append(("kill", worker.name)))

    wrapper.close()
    wrapper.close()

    assert events == [
        ("close", "a"),
        ("close", "b"),
        ("get", ("a", "b")),
        ("kill", "a"),
        ("kill", "b"),
    ]
    assert wrapper._workers == []
```

Extend the existing `run_opsd` failure test to inspect the `runtime_env.env_vars` passed to fake `ray.init` and assert it contains the monkeypatched `AGENTOPSD_RUN_ID`.

- [ ] **Step 2: Run runtime tests and verify the new expectations fail**

Run:

```bash
pytest -q tests/test_agentopsd_cleanup.py tests/test_main_opsd_device_setup.py
```

Expected: failures show obsolete preflight ownership and incomplete WebShop idempotence/run-id assertions; no Ray cluster starts because all Ray entry points are monkeypatched.

- [ ] **Step 3: Narrow `verl.trainer.agentopsd_cleanup` to owned runtime resources**

Retain `RUN_ID_ENV = "AGENTOPSD_RUN_ID"`, `_actor_key()`, `_iter_worker_handles()`, `_iter_environment_objects()`, and `cleanup_runtime_resources()`. Remove `argparse`, `/proc` inventory, preflight, signal, state-file, and CLI code from this training module.

The cleanup order remains:

```python
for environment in _iter_environment_objects(envs):
    environment.close()
for worker in _iter_worker_handles(trainer):
    ray_module.kill(worker, no_restart=True)
for pool in trainer.resource_pool_manager.resource_pool_dict.values():
    for placement_group in pool.pgs:
        remove_placement_group(placement_group)
```

Each operation catches and records its own exception so teardown cannot replace the original training exception. Deduplicate actor handles by actor id and environment objects by object identity.

- [ ] **Step 4: Complete `main_opsd` graceful cleanup and run-id propagation**

Preserve the existing device setup and training configuration changes. Ensure `run_opsd()`:

- sets a fallback run id only when the shell did not provide one;
- merges `AGENTOPSD_RUN_ID` into Ray `runtime_env.env_vars` without dropping existing variables;
- kills `OPSDTaskRunner` with `no_restart=True` in `finally`;
- calls `ray.shutdown()` only if this invocation initialized Ray.

Ensure `OPSDTaskRunner.run()` tracks trainer/train-env/val-env references and always invokes `cleanup_runtime_resources()` in `finally`, including failures during `trainer.init_workers()`, environment construction, or `trainer.fit()`.

- [ ] **Step 5: Make ALFWorld and WebShop actor teardown explicit and idempotent**

For both Ray-backed environment classes:

- initialize the worker list before actor creation so partial construction can be cleaned;
- pass `runtime_env={"env_vars": {"AGENTOPSD_RUN_ID": run_id}}` in actor options when the variable exists;
- implement actor-level `close()` that delegates to the held Gym environment;
- collect close futures, wait at most 5 seconds, then kill every actor with `no_restart=True`;
- tolerate already-dead actors, clear the worker list, set `_closed = True`, and return immediately on repeated close.

Use `_workers` for WebShop and `workers` for ALFWorld to preserve their existing public shapes. Do not modify Search beyond relying on its existing idempotent close of child environments, thread executor, and event loop.

- [ ] **Step 6: Run runtime and environment-focused tests**

Run:

```bash
pytest -q tests/test_agentopsd_cleanup.py tests/test_main_opsd_device_setup.py tests/test_agentopsd_local_assets.py
```

Expected: all tests pass without allocating GPUs or starting a Ray cluster.

- [ ] **Step 7: Checkpoint Task 4 conservatively**

Run:

```bash
git diff --check -- verl/trainer/agentopsd_cleanup.py verl/trainer/main_opsd.py agent_system/environments/env_package/alfworld/envs.py agent_system/environments/env_package/webshop/envs.py tests/test_agentopsd_cleanup.py tests/test_main_opsd_device_setup.py
git status --short
```

These paths were already dirty when the plan was written. Do not use a blanket `git add`. Compare against the initial worktree snapshot and stage only cleanup-specific hunks if they can be isolated without rewriting user work. Commit isolated hunks with:

```bash
git commit -m "fix: release AgentOPSD runtime resources"
```

If isolation is not reliable, keep the tested changes in the working tree and report that no Task 4 commit was created.

---

### Task 5: Full CPU-Only Verification and Requirement Audit

**Files:**
- Verify all files listed in the File Map.
- Modify only files implicated by a failing focused test.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: verified launcher preflight/exit behavior and graceful runtime cleanup.

- [ ] **Step 1: Run Python and Bash syntax checks**

Run:

```bash
python -m py_compile examples/process_cleanup/agentopsd_process_cleanup.py verl/trainer/agentopsd_cleanup.py verl/trainer/main_opsd.py
bash -n examples/process_cleanup/agentopsd_process_cleanup.sh examples/agentopsd_trainer/*.sh
```

Expected: both commands exit 0 with no output.

- [ ] **Step 2: Run the full focused regression suite**

Run:

```bash
pytest -q tests/test_agentopsd_process_cleanup.py tests/test_agentopsd_launcher_cleanup.py tests/test_agentopsd_cleanup.py tests/test_main_opsd_device_setup.py tests/test_agentopsd_local_assets.py
```

Expected: all tests pass; no test starts Ray, training, inference, or an environment simulation.

- [ ] **Step 3: Verify all launchers and forbidden broad-cleanup commands**

Run:

```bash
rg -L "agentopsd_cleanup_setup" examples/agentopsd_trainer/*.sh
rg -n "ray stop|pkill|killpg|verl\.trainer\.agentopsd_cleanup (preflight|cleanup)" examples/process_cleanup examples/agentopsd_trainer
```

Expected: both commands produce no matches. The first command's success condition is empty output; account for the installed ripgrep version's exit status by judging matches, not status alone.

- [ ] **Step 4: Audit the implementation against the design**

Confirm each item with code or test evidence:

1. New launch replaces a validated old launcher and exact `main_opsd` driver.
2. Recursive descendants, old-run actors, and legacy AgentOPSD actor markers are selected.
3. UID, repository, infrastructure, zombie, current launcher, and current cleaner exclusions are enforced.
4. PID start time is checked before TERM and KILL.
5. State writes are atomic, locked, mode-restricted, and repository-namespaced.
6. A superseded preflight cannot launch and cannot erase the newer state.
7. All seven scripts use one common hook and preserve exit status.
8. Main, train/validation environments, Ray workers, placement groups, and locally initialized Ray are cleaned gracefully.

- [ ] **Step 5: Inspect the final diff without launching any resource-consuming task**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` exits 0. Report all remaining pre-existing and newly modified paths separately; do not claim ownership of unrelated changes and do not run any `.sh` training launcher as verification.
