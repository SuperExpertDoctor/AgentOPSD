# AgentOPSD residual process cleanup design

## Goal

Add a reusable cleanup mechanism under `examples` and integrate it into every
shell launcher in `examples/agentopsd_trainer`. Starting a launcher must replace
an older AgentOPSD run from the same repository: it terminates the old driver,
its environment processes, and its training workers before starting the new
driver.

Cleanup must not terminate another user's processes, processes belonging to a
different checkout, or shared Ray infrastructure such as `raylet`,
`gcs_server`, the object store, or the Ray dashboard.

## Components

### Python process cleaner

`examples/process_cleanup/agentopsd_process_cleanup.py` provides a small CLI
and independently testable process-selection functions. It reads process
metadata from `/proc`, including PID, parent PID, process group, owner, working
directory, command line, environment run id, and process start time.

It exposes two operations:

- `preflight`: find and terminate older AgentOPSD runs before a new launch.
- `cleanup`: terminate processes that belong to the current run during shell
  exit handling.

The cleaner uses only the Python standard library so that it can run before Ray
or the training stack is initialized.

### Shell launcher hook

`examples/process_cleanup/agentopsd_process_cleanup.sh` defines the common
launcher setup and exit trap. Each script under `examples/agentopsd_trainer`
sources this file and invokes one setup function before data preparation or
training begins.

The setup function:

1. Resolves the repository root and launcher path.
2. Creates and exports a unique `AGENTOPSD_RUN_ID` and launcher PID.
3. Under a short repository-scoped lock, reads the previous run record, takes
   a process snapshot, and atomically publishes the new run claim.
4. Releases the lock and terminates the old launcher, drivers, descendants,
   environment workers, and training workers selected from that snapshot.
5. Takes a fresh process snapshot to catch children spawned while the old
   launcher was stopping, and terminates any remaining old-run processes.
6. Reacquires the lock and verifies that the run record still names this run;
   otherwise a newer launcher has replaced it and this launcher exits.
7. Refuses to continue if cleanup or final ownership validation fails.
8. Installs an exit trap that cleans the current run while preserving the
   launcher's original exit status.

The lock protects each state transition and its associated snapshot, but is not
held while waiting for processes to exit. The newest launcher wins by replacing
the state claim. Every launcher revalidates ownership before returning from
preflight, so a launcher superseded during cleanup cannot proceed to training.
This also lets the replaced launcher's exit trap run without deadlocking on a
lock held by the replacement.

### In-process resource cleanup

The AgentOPSD entry point propagates `AGENTOPSD_RUN_ID` through Ray's runtime
environment. Normal and exceptional exits explicitly close train and validation
environments, kill owned Ray actor handles, remove owned placement groups, and
shut down a Ray client initialized by the entry point. This is the graceful
path; the launcher hook remains the crash and restart fallback.

Environment implementations must provide idempotent `close()` behavior where
they own executors, subprocesses, or Ray actors.

## Process ownership and selection

A process is eligible only when its effective owner is the current user.
Preflight then builds the cleanup set from:

- the previous launcher recorded for this repository, after validating its PID
  and process start time, plus its recursive descendants;
- live `python -m verl.trainer.main_opsd` drivers associated with the current
  repository;
- recursive descendants of those drivers;
- non-infrastructure processes carrying a previous `AGENTOPSD_RUN_ID` from the
  repository's run record;
- known legacy AgentOPSD Ray actors associated with the repository, for runs
  created before run-id propagation existed.

Repository association requires a matching resolved working directory or an
unambiguous repository path in the command line. Marker matching alone is not
enough for drivers. Legacy worker matching is limited to exact AgentOPSD actor
class markers and the repository association check.

The cleaner explicitly excludes itself, the new launcher, zombie processes,
and Ray infrastructure. A validated older launcher is a target so that a
second invocation can replace a script that is still preparing data and has
not started `main_opsd` yet. The cleaner never uses `ray stop`, broad `pkill`,
or process-group termination because a process group or Ray cluster may be
shared.

## Termination protocol

The cleaner sends `SIGTERM` to selected old drivers and a validated old
launcher first to allow their normal cleanup handlers to run. It waits for a
bounded grace period, refreshes process state, then sends `SIGTERM` to any
remaining selected descendants and workers. After a second bounded wait it
sends `SIGKILL` only to survivors whose owner, start time, command, working
directory, and run id still match the captured record.

Rechecking identity before every destructive signal prevents a recycled PID
from being targeted. Failure to inspect or terminate a selected process makes
preflight fail; training is not launched with a partially cleaned prior run.

## Run state

Runtime state is stored below `${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}` in a
per-user, per-repository namespace. The repository namespace is derived from a
stable digest of the resolved repository root. State contains the run id,
launcher PID, launcher process start time, launcher path, repository root, and
creation timestamp. It must not contain credentials or the full training
command.

State writes use a temporary file and atomic rename. Exit cleanup removes the
state under the same lock only when it still names the exiting run. A failed
preflight also removes only its own claim; it cannot erase a newer launcher's
state.

## Shell integration

All seven current launchers in `examples/agentopsd_trainer` use the shared
hook. Their training arguments, selected Python interpreter, environment
variables, and data preparation behavior remain otherwise unchanged.

The common setup accepts the script's selected Python executable rather than
hard-coding a Conda environment. The hook must work when the launcher is called
from any current directory.

## Errors and observability

The cleaner prints concise records of discovered and terminated PIDs without
printing full commands or process environments. Preflight exits nonzero when
locking, state validation, inspection, or required termination fails. The shell
hook propagates that failure and does not launch training.

Exit cleanup is best effort so that it does not replace the training process's
original exit code, but it reports any surviving process IDs to stderr.

## Tests

CPU-only tests cover:

- automatic termination of an old repository-local driver;
- recursive descendant and run-id worker selection;
- legacy AgentOPSD worker selection;
- preservation of other users' and other repositories' processes;
- preservation of Ray infrastructure;
- exclusion of the current cleaner and launcher;
- PID reuse protection using process start time;
- escalation from `SIGTERM` to `SIGKILL`;
- preflight failure when a selected process survives;
- run-state validation and atomic replacement;
- source/setup integration in all AgentOPSD launcher scripts;
- preservation of a launcher's original exit status in the shell trap.

Tests use synthetic process records and temporary state directories. They do
not start Ray, environments, training, or GPU work.

## Out of scope

This change does not stop a shared Ray cluster, clean jobs belonging to other
repositories or users, register or launch training, or retrofit launchers
outside `examples/agentopsd_trainer`. The reusable hook is designed so those
other example families can opt in later.
