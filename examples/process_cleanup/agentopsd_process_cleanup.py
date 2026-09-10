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
