# Task 3 Report: Device as the AgentOPSD GPU Selector

## Scope and Implementation

Implemented Task 3 in the isolated `agentopsd-device-selection` checkout. No
AgentOPSD launch script, Ray process, training, evaluation, simulation,
rendering, or other resource-consuming algorithm was invoked.

Changes made:

- Updated `verl/trainer/config/agentopsd_trainer.yaml` to describe
  `trainer.device` as the sole user-facing selector. It retains
  `n_gpus_per_node: 1` as the compatibility field derived by
  `configure_training_devices`.
- Removed the sole `trainer.n_gpus_per_node` override from each of the seven
  AgentOPSD launch scripts. The final `trainer.device` override is now the only
  user device selection in each script.
- Replaced the static configuration test with the `device` selector contract
  and asserted that every AgentOPSD script has no independent
  `trainer.n_gpus_per_node` override, continues forwarding `$@`, and retains
  the rollout memory setting.
- Added README guidance for `trainer.device=[1,5]` and `trainer.device=cuda`.
  The documentation explicitly instructs users not to pass
  `trainer.n_gpus_per_node` and does not document `CUDA_VISIBLE_DEVICES` as a
  selection mechanism.

## TDD Evidence

### RED

After replacing the static test assertions and before changing config, scripts,
or README, ran:

```bash
conda run -n llm_ft_py310 pytest -q tests/test_agentopsd_local_assets.py
```

Observed output:

```text
....F.                                                                   [100%]
FAILED tests/test_agentopsd_local_assets.py::test_agentopsd_scripts_use_local_models_and_datasets
AssertionError: assert 'trainer.n_gpus_per_node' not in source
'trainer.n_gpus_per_node' is contained here:
    trainer.n_gpus_per_node=8 \\
1 failed, 5 passed, 1 warning in 4.14s
```

The failure was expected and demonstrated that the scripts still had independent
GPU-count overrides.

### GREEN

After applying the scoped implementation, ran:

```bash
conda run -n llm_ft_py310 pytest -q tests/test_agentopsd_local_assets.py
```

Observed output:

```text
......                                                                   [100%]
6 passed, 1 warning in 4.13s
```

Both pytest runs emitted this non-failing environment warning:

```text
Warning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

## Final CPU-Only Verification

Ran:

```bash
conda run -n llm_ft_py310 pytest -q \
  tests/test_device_utils.py \
  tests/test_main_opsd_device_setup.py \
  tests/test_agentopsd_local_assets.py
git diff --check
```

Observed output:

```text
.......................                                                  [100%]
23 passed, 1 warning in 4.42s
```

`git diff --check` produced no output and exited successfully.

## Changed Files

- `README.md`
- `verl/trainer/config/agentopsd_trainer.yaml`
- `examples/agentopsd_trainer/run_alfworld_3b.sh`
- `examples/agentopsd_trainer/run_alfworld_7b.sh`
- `examples/agentopsd_trainer/run_alfworld_7b_gpu1.sh`
- `examples/agentopsd_trainer/run_search_3b.sh`
- `examples/agentopsd_trainer/run_search_7b.sh`
- `examples/agentopsd_trainer/run_webshop_3b.sh`
- `examples/agentopsd_trainer/run_webshop_7b.sh`
- `tests/test_agentopsd_local_assets.py`
- `.superpowers/sdd/task-3-report.md`

## Self-Review

- Confirmed the change is confined to Task 3 files and does not modify Task 1
  or Task 2 implementation files.
- Confirmed all seven scripts retain their expected `trainer.device` form,
  `$@` forwarding, and `gpu_memory_utilization=0.85` setting.
- Confirmed no `trainer.n_gpus_per_node` string remains under
  `examples/agentopsd_trainer`.
- Confirmed no `CUDA_VISIBLE_DEVICES` guidance was added to the README.
- Inspected the final diff and verified whitespace with `git diff --check`.

## Concerns

- Pytest reports a third-party PyTorch `pynvml` deprecation warning. It is
  unrelated to Task 3 and does not fail the CPU/static test suite.
- Runtime GPU/Ray validation was intentionally not performed because the task
  explicitly restricts validation to CPU/static tests and forbids invoking
  launch scripts or resource-consuming workloads.
