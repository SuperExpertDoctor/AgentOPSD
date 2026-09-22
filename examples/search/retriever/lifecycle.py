"""Run a registered retrieval service only while its training owner is alive."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REGISTRY = Path('/media/abc_disk/admin123/lbh/dengji.txt')


def process_identity(pid):
    directory = Path('/proc') / str(pid)
    fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ProcessLookupError(pid)
    return directory.stat().st_uid, int(fields[19])


def owner_alive(pid, identity):
    try:
        return process_identity(pid) == identity
    except (OSError, ValueError):
        return False


def supervise(command, owner_pid, identity, on_started, *, grace=5, interval=1):
    if not owner_alive(owner_pid, identity):
        raise RuntimeError('training owner is no longer alive')
    requested = []
    previous = {}
    child = None
    reason = 'service exited'
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda signum, frame: requested.append(signum))
        child = subprocess.Popen(command, start_new_session=True)
        on_started(child)
        while child.poll() is None:
            if requested:
                reason = 'supervisor received signal'
                break
            if not owner_alive(owner_pid, identity):
                reason = 'owner exited'
                break
            time.sleep(interval)
    finally:
        if child is not None:
            # The child owns a new process group; never signal the training group.
            # Also clean descendants if the root service has already exited.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return child.returncode, reason


def latest_event(job_id):
    latest = None
    with REGISTRY.open('rb') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue  # Older records in this shared log can use another encoding.
            if event.get('job_id') == job_id:
                latest = event
    return latest


def append_event(job_id, event, **fields):
    record = dict(job_id=job_id, event=event, status=event, **fields)
    with REGISTRY.open('ab') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    if latest_event(job_id) != record:
        raise RuntimeError('registration readback failed')


def now():
    return datetime.datetime.now().astimezone().isoformat()


def stop_training_owner(owner_pid, identity):
    """Interrupt the foreground driver too: bash defers traps while waiting."""
    if not owner_alive(owner_pid, identity):
        return
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from examples.process_cleanup.agentopsd_process_cleanup import read_process, terminate_processes
    children = Path(f'/proc/{owner_pid}/task/{owner_pid}/children')
    try:
        pids = [int(value) for value in children.read_text().split()]
    except FileNotFoundError:
        return
    drivers = []
    for pid in pids:
        child = read_process(pid, expected_uid=identity[0])
        if child is not None and child.ppid == owner_pid and any(
                child.cmdline[i:i+2] == ('-m', 'verl.trainer.main_opsd')
                for i in range(len(child.cmdline)-1)):
            drivers.append(child)
    # Queue the shell's trap first, then release its foreground wait. Sending
    # TERM afterwards could interrupt the EXIT cleanup that has just begun.
    if owner_alive(owner_pid, identity):
        os.kill(owner_pid, signal.SIGTERM)
    terminate_processes(drivers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    registration = latest_event(args.job_id)
    if not registration or registration.get('status') != 'REGISTERED':
        parser.error('job must have a validated REGISTERED event before launch')
    if not command:
        parser.error('service command is required')
    try:
        identity = process_identity(args.owner_pid)
    except (OSError, ValueError):
        append_event(args.job_id, 'CANCELLED', ended_at=now(), exit_code=1,
                     reason='Training owner exited before service launch')
        return 1
    if identity[0] != os.getuid():
        append_event(args.job_id, 'FAILED', ended_at=now(), exit_code=1,
                     reason='Training owner uid mismatch; service was not started')
        return 1
    expected_owner = registration.get('owner_identity')
    if expected_owner != {'pid': args.owner_pid, 'uid': identity[0], 'start_time': identity[1]}:
        append_event(args.job_id, 'FAILED', ended_at=now(), exit_code=1,
                     reason='Training owner identity mismatch; service was not started')
        return 1

    def started(child):
        pid = os.getpid()
        append_event(args.job_id, 'RUNNING', started_at=now(),
                     root_pid=pid, pgid=os.getpgid(pid), worker_pids=[child.pid],
                     worker_process_groups={str(child.pid): os.getpgid(child.pid)},
                     actual_gpu_ids=registration['gpu_request']['ids'],
                     owner_identity=expected_owner, log_path=registration['log_path'])
        if process_identity(child.pid)[0] != os.getuid() or os.getpgid(child.pid) != child.pid:
            raise RuntimeError('invalid service process binding')
        if int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[1]) != pid:
            raise RuntimeError('service parent does not match supervisor')
        print(f'registered service pid={child.pid} owner={args.owner_pid}', flush=True)

    try:
        code, reason = supervise(command, args.owner_pid, identity, started)
    except BaseException:
        append_event(args.job_id, 'FAILED', ended_at=now(), exit_code=1,
                     reason='Supervisor error; owned service was cleaned up', result_path=registration['log_path'])
        raise
    status = 'CANCELLED' if reason != 'service exited' else ('SUCCEEDED' if code == 0 else 'FAILED')
    append_event(args.job_id, status, ended_at=now(), exit_code=code,
                 reason=reason, result_path=registration['log_path'])
    if (reason == 'service exited' and os.environ.get('RETRIEVAL_STOP_OWNER_ON_FAILURE') == '1'
            and owner_alive(args.owner_pid, identity)):
        print('Retrieval dependency exited; stopping its training launcher.', file=sys.stderr, flush=True)
        stop_training_owner(args.owner_pid, identity)
    return 0 if status == 'CANCELLED' else (code if code >= 0 else 128 - code)


if __name__ == '__main__':
    sys.exit(main())
