# Retrieval startup and lifecycle

The AgentOPSD 3B and 7B search launchers now manage their local retrieval service:

```bash
bash examples/agentopsd_trainer/run_search_3b.sh
```

Before invoking the trainer, the launcher:

1. Checks the local endpoint, model/index paths, free GPU memory and available RAM.
2. Starts an owned supervisor and records the actual service PID in a private
   temporary directory.
3. Waits for `/health` to identify the expected service process, then requires a
   real `/retrieve` warmup request to succeed.
4. Starts training only after readiness. Startup failures stop the launcher with
   a diagnostic message and a log path, rather than entering rollout retries.

To verify the entire retrieval startup and cleanup path without training:

```bash
RETRIEVAL_CHECK_ONLY=1 bash examples/agentopsd_trainer/run_search_3b.sh
```

Defaults are `http://127.0.0.1:8081/retrieve`, GPU 6 (GPU 1 in the 3B
Search launcher), and a 1200-second startup
budget (large index/corpus loading is separate from the 60-second search timeout).
`RETRIEVAL_GPU_ID`, `RETRIEVAL_PYTHON`, and `RETRIEVAL_STARTUP_TIMEOUT` override
these settings. The launchers provide `ASSET_DATA_DIR` and `ASSET_WEIGHTS_DIR`
for the index, corpus, and encoder model. `SEARCH_URL`, or the final `env.search.search_url=...` argument,
can select another local port. Managed URLs must use `127.0.0.1` or `localhost`
and `/retrieve`. An occupied port fails startup; unrelated services are never
adopted or killed. The search client concurrency is 8 by default in both launchers.

On normal exit, Ctrl-C or SIGTERM, the launcher stops its retrieval supervisor
before generic training cleanup.
The supervisor also watches the launcher's UID and process start time, so a
launcher killed without running its traps cannot leave the service behind.
It terminates the service's dedicated process group and escalates after five
seconds if needed. If the retrieval process exits unexpectedly, it interrupts
the foreground training driver and signals the launcher to run cleanup.

For a separately managed service, `retrieval_launch.sh` requires
`RETRIEVAL_OWNER_PID`, `RETRIEVAL_OWNER_START_TIME`, and
`RETRIEVAL_SERVICE_PID_FILE`. Do not bind it to an
interactive login shell.

The server logs `encode`, `index_search`, and `load_docs` stage times with a
request ID. Startup separately logs `load_index` and `load_corpus`. Query text
and credentials are omitted. SIGUSR1 sent to the service PID prints all Python
thread stacks to stderr without restarting the service.
