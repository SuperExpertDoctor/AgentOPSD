from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
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
        arg == "-m"
        and index + 1 < len(process.cmdline)
        and process.cmdline[index + 1] == module
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
        start_time = _parse_start_time(
            (directory / "stat").read_text(encoding="utf-8", errors="replace")
        )
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


def _inspect_for_termination(reader, pid: int) -> ProcessInfo | None:
    try:
        return reader(pid)
    except CleanupError:
        raise
    except ProcessInspectionError:
        raise
    except (PermissionError, OSError) as exc:
        raise CleanupError(f"cannot revalidate pid {pid}: {exc}") from exc


def _signal_process(send_signal, pid: int, sig: signal.Signals) -> bool:
    try:
        send_signal(pid, sig)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError) as exc:
        raise CleanupError(f"cannot signal pid {pid}: {exc}") from exc
    return True


def terminate_processes(
    processes: Sequence[ProcessInfo],
    *,
    reader=read_process,
    send_signal=os.kill,
    grace_period: float = 2.0,
) -> TerminationResult:
    """Terminate only captured processes whose identity still matches."""

    captured = {item.pid: item for item in processes}
    terminated = []
    for item in captured.values():
        if same_process(_inspect_for_termination(reader, item.pid), item):
            if _signal_process(send_signal, item.pid, signal.SIGTERM):
                terminated.append(item.pid)

    deadline = time.monotonic() + max(0.0, grace_period)
    survivors = [
        item
        for item in captured.values()
        if same_process(_inspect_for_termination(reader, item.pid), item)
    ]
    while survivors and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        survivors = [
            item
            for item in survivors
            if same_process(_inspect_for_termination(reader, item.pid), item)
        ]

    killed = []
    for item in survivors:
        if same_process(_inspect_for_termination(reader, item.pid), item):
            if _signal_process(send_signal, item.pid, signal.SIGKILL):
                killed.append(item.pid)

    kill_deadline = time.monotonic() + max(0.0, grace_period)
    final = [
        item
        for item in survivors
        if same_process(_inspect_for_termination(reader, item.pid), item)
    ]
    while final and time.monotonic() < kill_deadline:
        time.sleep(min(0.05, max(0.0, kill_deadline - time.monotonic())))
        final = [
            item
            for item in final
            if same_process(_inspect_for_termination(reader, item.pid), item)
        ]

    return TerminationResult(
        tuple(terminated),
        tuple(killed),
        tuple(item.pid for item in final),
    )


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


def state_paths(repo_root, *, runtime_root=None, uid=None) -> StatePaths:
    resolved = os.path.realpath(repo_root)
    owner = os.getuid() if uid is None else uid
    base = Path(
        runtime_root
        or os.environ.get("XDG_RUNTIME_DIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    directory = base / f"agentopsd-{owner}" / digest
    return StatePaths(directory, directory / "run.json", directory / "run.lock")


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


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _load_state_path(path: Path, repo_root: str) -> RunState | None:
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

    if not isinstance(state.run_id, str) or not state.run_id:
        raise CleanupError("invalid AgentOPSD run state: run_id")
    if not _is_positive_int(state.launcher_pid):
        raise CleanupError("invalid AgentOPSD run state: launcher_pid")
    if not _is_positive_int(state.launcher_start_time):
        raise CleanupError("invalid AgentOPSD run state: launcher_start_time")
    if not isinstance(state.launcher_path, str) or not state.launcher_path:
        raise CleanupError("invalid AgentOPSD run state: launcher_path")
    if not isinstance(state.repo_root, str) or not state.repo_root:
        raise CleanupError("invalid AgentOPSD run state: repo_root")
    if not isinstance(state.created_at, str) or not state.created_at:
        raise CleanupError("invalid AgentOPSD run state: created_at")
    if os.path.realpath(state.repo_root) != os.path.realpath(repo_root):
        raise CleanupError("invalid AgentOPSD run state: repository root")
    return state


def _write_state_path(path: Path, state: RunState) -> None:
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


def write_state(state: RunState, *, runtime_root=None) -> None:
    with locked_state(state.repo_root, runtime_root=runtime_root) as paths:
        _write_state_path(paths.state, state)


def _snapshot(process_provider) -> tuple[ProcessInfo, ...]:
    try:
        return tuple(process_provider())
    except CleanupError:
        raise
    except ProcessInspectionError:
        raise
    except (PermissionError, OSError) as exc:
        raise CleanupError(f"cannot inspect process table: {exc}") from exc


def _command_mentions_path(process: ProcessInfo, path: str) -> bool:
    expected = os.path.realpath(path)
    for argument in process.cmdline:
        if argument.startswith("-"):
            continue
        candidate = argument
        if not os.path.isabs(candidate):
            candidate = os.path.join(process.cwd, candidate)
        if os.path.realpath(candidate) == expected:
            return True
    return False


def _validate_current_launcher(
    processes: Sequence[ProcessInfo],
    *,
    repo_root: str,
    launcher_path: str,
    launcher_pid: int,
    current_uid: int,
) -> ProcessInfo:
    candidate = next((item for item in processes if item.pid == launcher_pid), None)
    if candidate is None:
        raise CleanupError(f"cannot validate launcher pid {launcher_pid}")
    if candidate.uid != current_uid or candidate.state == "Z":
        raise CleanupError(f"cannot validate launcher pid {launcher_pid}")
    if not _belongs_to_repo(candidate, repo_root) or not _command_mentions_path(candidate, launcher_path):
        raise CleanupError(f"launcher pid {launcher_pid} is not the requested repository launcher")
    return candidate


def _ignored_pids(launcher_pid: int) -> set[int]:
    ignored = {os.getpid(), launcher_pid}
    return ignored


def _merge_termination_results(results: Sequence[TerminationResult]) -> TerminationResult:
    terminated = []
    killed = []
    survivors = []
    for result in results:
        terminated.extend(result.terminated)
        killed.extend(result.killed)
        survivors.extend(result.survivors)
    return TerminationResult(
        tuple(dict.fromkeys(terminated)),
        tuple(dict.fromkeys(killed)),
        tuple(dict.fromkeys(survivors)),
    )


def _raise_if_surviving(result: TerminationResult) -> None:
    if result.survivors:
        pids = ",".join(str(pid) for pid in result.survivors)
        raise CleanupError(f"selected processes survived cleanup: {pids}")


def _claim_matches(
    state: RunState | None,
    *,
    repo_root: str,
    launcher_path: str,
    launcher_pid: int,
    run_id: str,
) -> bool:
    return state is not None and (
        state.run_id == run_id
        and state.launcher_pid == launcher_pid
        and os.path.realpath(state.launcher_path) == os.path.realpath(launcher_path)
        and os.path.realpath(state.repo_root) == os.path.realpath(repo_root)
    )


def _require_claim(paths: StatePaths, *, repo_root: str, launcher_path: str, launcher_pid: int, run_id: str) -> RunState:
    state = _load_state_path(paths.state, repo_root)
    if not _claim_matches(
        state,
        repo_root=repo_root,
        launcher_path=launcher_path,
        launcher_pid=launcher_pid,
        run_id=run_id,
    ):
        raise CleanupError("preflight was replaced by a newer launcher")
    return state


def _discard_claim(
    repo_root: str,
    *,
    launcher_path: str,
    launcher_pid: int,
    run_id: str,
    runtime_root=None,
) -> None:
    try:
        with locked_state(repo_root, runtime_root=runtime_root) as paths:
            state = _load_state_path(paths.state, repo_root)
            if _claim_matches(
                state,
                repo_root=repo_root,
                launcher_path=launcher_path,
                launcher_pid=launcher_pid,
                run_id=run_id,
            ):
                try:
                    paths.state.unlink()
                except FileNotFoundError:
                    pass
    except (CleanupError, OSError):
        pass


def _refresh_targets(
    targets: Sequence[ProcessInfo], processes: Sequence[ProcessInfo]
) -> tuple[ProcessInfo, ...]:
    current = {item.pid: item for item in processes}
    return tuple(
        item
        for item in targets
        if same_process(current.get(item.pid), item)
    )


def _selection_results(
    selection: Selection,
    *,
    process_provider,
    require_claim=None,
) -> TerminationResult:
    results = []
    primary_result = terminate_processes(selection.primary)
    results.append(primary_result)
    _raise_if_surviving(primary_result)
    if require_claim is not None:
        require_claim()

    refreshed = _refresh_targets(selection.remaining, _snapshot(process_provider))
    remaining_result = terminate_processes(refreshed)
    results.append(remaining_result)
    _raise_if_surviving(remaining_result)
    if require_claim is not None:
        require_claim()
    return _merge_termination_results(results)


def preflight(
    *,
    repo_root: str,
    launcher_path: str,
    launcher_pid: int,
    run_id: str,
    runtime_root=None,
    process_provider=iter_processes,
) -> TerminationResult:
    """Claim a launcher slot and terminate stale processes from this repo."""

    if not isinstance(run_id, str) or not run_id:
        raise CleanupError("invalid run id")
    if not _is_positive_int(launcher_pid):
        raise CleanupError("invalid launcher pid")

    resolved_repo = os.path.realpath(repo_root)
    resolved_launcher = os.path.realpath(launcher_path)
    current_uid = os.getuid()
    claim_written = False
    previous = None
    previous_launcher = None
    try:
        with locked_state(resolved_repo, runtime_root=runtime_root) as paths:
            previous = _load_state_path(paths.state, resolved_repo)
            initial_snapshot = _snapshot(process_provider)
            current_launcher = _validate_current_launcher(
                initial_snapshot,
                repo_root=resolved_repo,
                launcher_path=resolved_launcher,
                launcher_pid=launcher_pid,
                current_uid=current_uid,
            )
            if previous is not None:
                previous_launcher = ProcessIdentity(
                    previous.launcher_pid,
                    previous.launcher_start_time,
                )
            state = RunState(
                run_id=run_id,
                launcher_pid=launcher_pid,
                launcher_start_time=current_launcher.start_time,
                launcher_path=resolved_launcher,
                repo_root=resolved_repo,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            )
            _write_state_path(paths.state, state)
            claim_written = True

        ignored = _ignored_pids(launcher_pid)
        previous_run_id = previous.run_id if previous is not None else None
        selection = select_preflight_targets(
            initial_snapshot,
            repo_root=resolved_repo,
            previous_run_id=previous_run_id,
            previous_launcher=previous_launcher,
            current_uid=current_uid,
            ignored_pids=ignored,
        )

        def require_claim():
            with locked_state(resolved_repo, runtime_root=runtime_root) as claim_paths:
                _require_claim(
                    claim_paths,
                    repo_root=resolved_repo,
                    launcher_path=resolved_launcher,
                    launcher_pid=launcher_pid,
                    run_id=run_id,
                )

        results = [_selection_results(
            selection,
            process_provider=process_provider,
            require_claim=require_claim,
        )]

        with locked_state(resolved_repo, runtime_root=runtime_root) as paths:
            _require_claim(
                paths,
                repo_root=resolved_repo,
                launcher_path=resolved_launcher,
                launcher_pid=launcher_pid,
                run_id=run_id,
            )
            second_snapshot = _snapshot(process_provider)

        late_selection = select_preflight_targets(
            second_snapshot,
            repo_root=resolved_repo,
            previous_run_id=previous_run_id,
            previous_launcher=previous_launcher,
            current_uid=current_uid,
            ignored_pids=ignored,
        )
        results.append(_selection_results(
            late_selection,
            process_provider=process_provider,
            require_claim=require_claim,
        ))

        with locked_state(resolved_repo, runtime_root=runtime_root) as paths:
            _require_claim(
                paths,
                repo_root=resolved_repo,
                launcher_path=resolved_launcher,
                launcher_pid=launcher_pid,
                run_id=run_id,
            )
        return _merge_termination_results(results)
    except (CleanupError, ProcessInspectionError):
        if claim_written:
            _discard_claim(
                resolved_repo,
                launcher_path=resolved_launcher,
                launcher_pid=launcher_pid,
                run_id=run_id,
                runtime_root=runtime_root,
            )
        raise
    except (PermissionError, OSError) as exc:
        if claim_written:
            _discard_claim(
                resolved_repo,
                launcher_path=resolved_launcher,
                launcher_pid=launcher_pid,
                run_id=run_id,
                runtime_root=runtime_root,
            )
        raise CleanupError(f"preflight failed: {exc}") from exc


def _select_run_targets(
    processes: Sequence[ProcessInfo],
    *,
    run_id: str,
    current_uid: int,
    ignored_pids: set[int],
) -> tuple[ProcessInfo, ...]:
    return tuple(
        sorted(
            (
                item
                for item in processes
                if item.uid == current_uid
                and item.state != "Z"
                and item.pid not in ignored_pids
                and item.run_id == run_id
                and not _is_infrastructure(item)
                and not _is_cleanup_process(item)
            ),
            key=lambda item: item.pid,
        )
    )


def _is_cleanup_process(process: ProcessInfo) -> bool:
    return "agentopsd_process_cleanup.py" in process.command


def cleanup_run(
    *,
    repo_root: str,
    launcher_path: str,
    launcher_pid: int,
    run_id: str,
    runtime_root=None,
    process_provider=iter_processes,
) -> TerminationResult:
    """Best-effort cleanup for one run, without touching a newer claim."""

    if not isinstance(run_id, str) or not run_id:
        raise CleanupError("invalid run id")
    resolved_repo = os.path.realpath(repo_root)
    resolved_launcher = os.path.realpath(launcher_path)
    ignored = _ignored_pids(launcher_pid)
    with locked_state(resolved_repo, runtime_root=runtime_root) as paths:
        state = _load_state_path(paths.state, resolved_repo)
    targets = _select_run_targets(
        _snapshot(process_provider),
        run_id=run_id,
        current_uid=os.getuid(),
        ignored_pids=ignored,
    )
    result = terminate_processes(targets)
    _raise_if_surviving(result)

    with locked_state(resolved_repo, runtime_root=runtime_root) as paths:
        current = _load_state_path(paths.state, resolved_repo)
        if _claim_matches(
            current,
            repo_root=resolved_repo,
            launcher_path=resolved_launcher,
            launcher_pid=launcher_pid,
            run_id=run_id,
        ):
            try:
                paths.state.unlink()
            except FileNotFoundError:
                pass
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean stale AgentOPSD processes.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "cleanup"):
        command = subparsers.add_parser(action)
        command.add_argument("--repo-root", required=True)
        command.add_argument("--launcher", required=True)
        command.add_argument("--launcher-pid", required=True, type=int)
        command.add_argument("--run-id", required=True)
    return parser


def _print_result(action: str, run_id: str, result: TerminationResult) -> None:
    pids = tuple(dict.fromkeys(result.terminated + result.killed + result.survivors))
    print(
        f"action={action} run_id={run_id} "
        f"terminated={len(result.terminated)} killed={len(result.killed)} "
        f"survivors={len(result.survivors)} pids={list(pids)}"
    )


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        operation = preflight if args.action == "preflight" else cleanup_run
        result = operation(
            repo_root=args.repo_root,
            launcher_path=args.launcher,
            launcher_pid=args.launcher_pid,
            run_id=args.run_id,
        )
        _print_result(args.action, args.run_id, result)
        return 0
    except (CleanupError, ProcessInspectionError, PermissionError, OSError) as exc:
        print(f"agentopsd cleanup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
