# Retrieval startup and lifecycle

The AgentOPSD 3B and 7B search launchers now manage their local retrieval service:

```bash
bash examples/agentopsd_trainer/run_search_3b.sh
```

Before invoking the trainer, the launcher:

1. Checks the local endpoint, model/index paths, free GPU memory and available RAM.
2. Registers a new retrieval job using the identity in the repository `AGENTS.md`,
   under an exclusive lock in `/media/abc_disk/admin123/lbh/dengji.txt`, and validates it.
3. Starts an owned supervisor and records/validates the actual service PID.
4. Waits for `/health` to identify the expected service process, then requires a
   real `/retrieve` warmup request to succeed.
5. Starts training only after readiness. Startup failures stop the launcher with
   a diagnostic message and a log path, rather than entering rollout retries.

The existing registration requirements for the training job itself still apply.
The automatic registration above covers the retrieval dependency.

To verify the entire retrieval startup and cleanup path without training:

```bash
RETRIEVAL_CHECK_ONLY=1 bash examples/agentopsd_trainer/run_search_3b.sh
```

Defaults are `http://127.0.0.1:8081/retrieve`, GPU 6, and a 1200-second startup
budget (large index/corpus loading is separate from the 60-second search timeout).
`RETRIEVAL_GPU_ID`, `RETRIEVAL_PYTHON`, and `RETRIEVAL_STARTUP_TIMEOUT` override
these settings. `SEARCH_URL`, or the final `env.search.search_url=...` argument,
can select another local port. Managed URLs must use `127.0.0.1` or `localhost`
and `/retrieve`. An occupied port fails startup; unrelated services are never
adopted or killed. The search client concurrency is 8 by default in both launchers.

On normal exit, Ctrl-C or SIGTERM, the launcher stops its retrieval supervisor
and verifies a terminal registration event before generic training cleanup.
The supervisor also watches the launcher's UID and process start time, so a
launcher killed without running its traps cannot leave the service behind.
It terminates the service's dedicated process group and escalates after five
seconds if needed. If the retrieval process exits unexpectedly, it interrupts
the foreground training driver and signals the launcher to run cleanup.

For a separately managed service, `retrieval_launch.sh` still accepts an
explicit, already registered `RETRIEVAL_JOB_ID` and `RETRIEVAL_OWNER_PID`.
Do not bind it to an interactive login shell or reuse job IDs on restart.

The server logs `encode`, `index_search`, and `load_docs` stage times with a
request ID. Startup separately logs `load_index` and `load_corpus`. Query text
and credentials are omitted. SIGUSR1 sent to the service PID prints all Python
thread stacks to stderr without restarting the service.
