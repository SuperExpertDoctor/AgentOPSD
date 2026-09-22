import os
import signal
import subprocess
import sys
import time


def test_service_is_killed_when_owner_exits_even_if_term_is_ignored(tmp_path):
    from examples.search.retriever.lifecycle import supervise

    owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    service = None
    try:
        from examples.search.retriever.lifecycle import process_identity
        identity = process_identity(owner.pid)
        marker = tmp_path / 'ready'
        def started(child):
            nonlocal service
            service = child
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert marker.exists()
            owner.terminate()
            owner.wait(timeout=5)
        code, reason = supervise(
            [sys.executable, '-c',
             'import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
             f'pathlib.Path({str(marker)!r}).touch(); time.sleep(60)'],
            owner.pid, identity, started, grace=.1, interval=.02,
        )
        assert code == -signal.SIGKILL
        assert reason == 'owner exited'
        assert service.poll() is not None
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait()
        if service is not None and service.poll() is None:
            service.kill()
            service.wait()


def test_pid_reuse_is_not_a_live_owner():
    from examples.search.retriever.lifecycle import owner_alive, process_identity
    uid, start = process_identity(os.getpid())
    assert owner_alive(os.getpid(), (uid, start))
    assert not owner_alive(os.getpid(), (uid, start + 1))


def test_refuses_to_start_for_reused_owner_pid(tmp_path):
    import pytest
    from examples.search.retriever.lifecycle import process_identity, supervise
    uid, start = process_identity(os.getpid())
    marker = tmp_path / 'must-not-exist'
    with pytest.raises(RuntimeError, match='owner'):
        supervise([sys.executable, '-c', f'open({str(marker)!r}, "w").close()'],
                  os.getpid(), (uid, start + 1), lambda child: None)
    assert not marker.exists()


def test_failed_running_registration_cleans_started_service():
    import pytest
    from examples.search.retriever.lifecycle import process_identity, supervise
    children = []
    def fail_binding(child):
        children.append(child)
        raise RuntimeError('registration failed')
    with pytest.raises(RuntimeError, match='registration failed'):
        supervise([sys.executable, '-c', 'import time; time.sleep(60)'],
                  os.getpid(), process_identity(os.getpid()), fail_binding, grace=.1)
    assert children[0].poll() is not None


def test_natural_exit_preserves_service_status():
    from examples.search.retriever.lifecycle import process_identity, supervise
    code, reason = supervise([sys.executable, '-c', 'raise SystemExit(17)'],
                             os.getpid(), process_identity(os.getpid()),
                             lambda child: None, interval=.01)
    assert (code, reason) == (17, 'service exited')


def test_dependency_failure_interrupts_foreground_training_driver(tmp_path):
    from examples.search.retriever.lifecycle import process_identity, stop_training_owner
    marker = tmp_path / 'driver'
    # The fake driver only sleeps, but its argv matches the training entrypoint.
    code = f'import os,time,pathlib; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)'
    cleaned = tmp_path / 'cleaned'
    env = {**os.environ, 'CLEANED': str(cleaned)}
    owner = subprocess.Popen(['bash', '-c',
                              """trap 'exit 143' TERM; trap 'sleep .1; echo done > "$CLEANED"' EXIT; "$@" """,
                              'bash', sys.executable, '-c', code, '-m', 'verl.trainer.main_opsd'], env=env)
    driver = None
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert marker.exists()
        driver = int(marker.read_text())
        identity = process_identity(owner.pid)
        stop_training_owner(owner.pid, identity)
        assert owner.wait(timeout=5) != 0
        assert cleaned.read_text().strip() == 'done'
        from pathlib import Path
        assert not Path(f'/proc/{driver}').exists()
    finally:
        if owner.poll() is None: owner.kill()
        owner.wait()
        if driver:
            try: os.kill(driver,signal.SIGKILL)
            except ProcessLookupError: pass
