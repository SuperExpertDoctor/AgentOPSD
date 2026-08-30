# AGENTS

## Registration Identity

- `real_name`: 侯国强
- `name_id`: `houguoqiang`

## Training task registration gate (v1.2)

Before starting or restarting training, fine-tuning, inference evaluation,
simulation, rendering, or any other task that clearly consumes GPU, VRAM, CPU,
or RAM, read and follow
`/media/abc_disk/admin123/lbh/训练任务AI极简登记说明_v1.2.txt` (or a newer
versioned replacement in that directory) and the v1.2 workflow below. The sole
fixed registration file is `/media/abc_disk/admin123/lbh/dengji.txt`.
This gate applies regardless of whether the task is launched by a shell script,
Python CLI, tmux/screen, a container, or an automation pipeline.

### Identity and preflight

- First check that the user has supplied a Chinese real name or an all-lowercase
  English real name. If the name is missing, ask only for the name and do not
  start the task. Do not ask the user to fill in training files, purpose, timing,
  resource estimates, or other registration fields.
- Preserve a Chinese name exactly in `real_name` and generate a no-space,
  all-lowercase English `name_id` automatically. If the user supplied an
  all-lowercase English name, it may be used as both the real name and `name_id`.
  Never use a nickname, the shared Linux account, or the repository owner
  directory as the real name or `name_id`; record `linux_user` separately.
- Once identity is valid, inspect the current working directory, training entry
  point, configuration files, datasets, models, launch arguments, and server
  resource state using read-only checks. Any command stored in an event must be
  redacted, and passwords, private keys, tokens, API keys, or other credentials
  must never be written to the log.
- All fields other than the user's real name are filled in automatically. Use
  evidence from scripts, configs, checkpoints, model weights, data volume,
  `nvidia-smi`, CPU/memory state, and historical logs to estimate resources and
  duration. Use ranges where precision is uncertain; use `unknown` when the
  evidence is insufficient and explain the uncertainty in `estimate_basis`.

### REGISTERED event

- Before every start or restart, append a new JSON Lines (`JSONL`) event with
  `event: "REGISTERED"` under an exclusive file lock. Append only: never
  overwrite, delete, or edit historical records.
- The event must use a unique `job_id` and include at least:
  `registered_at`, `real_name`, `name_id`, `linux_user`, `host`, `ai` (tool or
  session identifier), `task_name`, `purpose`, `workdir`, the training entry
  file, related configuration files, `training_files`, `dataset`, `model`,
  `command_redacted`, `gpu_request` (planned GPU IDs/count and VRAM estimate),
  CPU-core, RAM, disk-growth, and duration estimates, `estimate_basis`,
  `log_path`, `checkpoint_path`, `result_path` when known,
  `resource_snapshot`, and `status: "REGISTERED"`.
- Do not launch until the append succeeds and the corresponding `job_id` is
  read back and validated as parseable JSON with the correct `real_name`,
  `name_id`, and `REGISTERED` status. A write, lock, or validation failure is a
  hard stop.
- After successful registration and before launch, report the registration
  summary to the user, including the name, `name_id`, `job_id`, task, expected
  resources/time, and `/media/abc_disk/admin123/lbh/dengji.txt`.

### RUNNING event and process binding

- After launch, append a `RUNNING` event with the same `job_id`, the actual
  `started_at`, current GPU IDs, and `status: "RUNNING"`. A `REGISTERED` event
  without a valid process-bound `RUNNING` event is not running authorization.
- A single-process task must record `pid`; record `pgid` whenever available.
  For multi-process tasks, record the launcher/root PID, process-group ID, and
  a pure JSON array of known worker PIDs, preferably as `worker_pids`.
- Accepted scalar PID fields are `pid`, `root_pid`, `launcher_pid`,
  `container_pid`, and `host_pid`. Accepted PID-array fields are `pids`,
  `process_ids`, `worker_pids`, `child_pids`, `active_worker_pids`,
  `container_pids`, and `host_pids`. Accepted process-group fields are `pgid`,
  `process_group`, and `process_group_id`. PID arrays must contain only positive
  integer values; do not encode them as objects, dictionaries, strings, or
  prose.
- If worker names or ranks are useful, put them in a separate structured field
  such as `worker_roles`; it cannot replace the checker-readable PID array.
- For Docker or another PID namespace, prefer host PIDs and also record
  `container_id`/`docker_container_id`; multiple container IDs belong in the
  corresponding `container_ids` or `docker_container_ids` array.
- After launch and after every restart or worker replacement, read back and
  validate the newest `RUNNING` event. Using read-only checks, verify
  `/proc/<pid>`, process ownership, parent/child relationships, process groups,
  and GPU processes so that every binding is a currently running process and not
  an exited or reused PID. If any event, PID, process-owner, relationship,
  process-group, or GPU validation fails, do not continue starting or restarting
  the task and do not claim that it has running authorization.
- If the parent exits, workers are reparented or regenerated, or training is
  restarted in tmux/screen or another supervisor, append a new `RUNNING` event
  immediately. Include `updated_at` or a new `started_at`, the current PID/PGID,
  current worker PID array, GPU IDs, and log path. Never overwrite the old event.
- If an already-running task is discovered without prior registration, do not
  claim that it satisfied the prior-registration-before-start rule. If an
  administrator chooses to retain it, append `retroactive_registration: true`
  in both the corrective `REGISTERED` and `RUNNING` records, recording the
  actual start time, discovery time, current PID/PGID, and reason. This protects
  only the confirmed current process and does not erase the prior violation.

### Terminal event

When the task ends or is stopped, append a final event with the same `job_id`:
`SUCCEEDED`, `FAILED`, `CANCELLED`, or `EXPIRED`. Include `ended_at`, the exit
code, the result path when applicable, and a redacted reason for failure,
cancellation, or timeout. `exit_code` is mandatory for `FAILED`, `CANCELLED`,
and `EXPIRED` events; use `unknown` only when the runtime provides no exit code.
Do not report the task as fully registered or complete until the latest event
has been read back and validated.

This is a strict "register first, run second, bind the process, and close the
loop" gate: every resource-consuming algorithm task must complete registration
and read-back validation before launch, maintain real process-binding evidence
during execution, and append a terminal event at the end. The event log is
append-only; historical records must never be modified.
