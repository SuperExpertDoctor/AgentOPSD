# Task 1 Report: Deterministic Device Selection

## Status

Implemented Task 1 in the isolated checkout:

`/home/shuixia/users/houguoqiang/code/opsd/AgentOPSD/.worktrees/agentopsd-device-selection`

No training, inference evaluation, simulation, rendering, Ray, or GPU-consuming
algorithm was launched. Tests were run CPU-only in conda environment
`llm_ft_py310`.

## Implementation Details

- Added the `requested_device` field to the frozen `DeviceSelection` contract.
- Made explicit GPU ID lists authoritative, normalized to string IDs, validated
  for non-empty, unique, non-negative integer values, and checked against
  injected accessible IDs.
- Made `trainer.device: cuda` select every accessible GPU in the supplied stable
  order rather than sampling the legacy `n_gpus_per_node` count.
- Derived and wrote back `trainer_config.n_gpus_per_node` from the resolved
  single-node selection.
- Rejected unsupported device backends and multi-node explicit GPU lists.
- Added runtime CUDA binding validation when accessible IDs are discovered from
  the host rather than injected by a test or caller.
- Added tensor-parallel validation for positive TP size, sufficient GPU count,
  and divisibility.

## TDD Evidence

### RED

After replacing `tests/test_device_utils.py` and before changing production
code, ran:

```text
conda run -n llm_ft_py310 pytest -q tests/test_device_utils.py
```

Exact relevant output:

```text
ERROR: found no collectors for /home/shuixia/users/houguoqiang/code/opsd/AgentOPSD/.worktrees/agentopsd-device-selection/tests/test_device_utils.py

ImportError: cannot import name 'validate_tensor_parallel_size' from 'verl.trainer.device_utils'
1 warning, 1 error in 4.40s
```

The RED process exited with code `4`, and the missing public API was the
expected failure against the baseline resolver.

RED registration lifecycle was recorded and validated in
`/media/abc_disk/admin123/lbh/dengji.txt`:

`20260901-140000-houguoqiang-device-utils-red-001`

The `RUNNING` event was bound to a live process and process group owned by the
current user. The terminal event is `FAILED` with exit code `4`, representing
the intentional RED phase.

### GREEN

After replacing `verl/trainer/device_utils.py` with the deterministic resolver,
ran:

```text
conda run -n llm_ft_py310 pytest -q tests/test_device_utils.py
```

Exact output:

```text
................                                                         [100%]
=============================== warnings summary ===============================
.../site-packages/torch/cuda/__init__.py:61
  Warning: The pynvml package is deprecated. Please install nvidia-ml-py instead.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
16 passed, 1 warning in 4.33s
```

GREEN registration lifecycle was recorded and validated in
`/media/abc_disk/admin123/lbh/dengji.txt`:

`20260901-140300-houguoqiang-device-utils-green-001`

The `RUNNING` event was bound to a live process and process group owned by the
current user. The terminal event is `SUCCEEDED` with exit code `0`.

## Changed Files

- `tests/test_device_utils.py`
- `verl/trainer/device_utils.py`
- `.superpowers/sdd/task-1-report.md`

## Self-Review

- Confirmed the public dataclass and function signatures match the task brief.
- Confirmed no `random` import, sampler, or `rng` argument remains.
- Confirmed the test suite covers explicit selection, deterministic CUDA
  selection, invalid lists, inaccessible IDs, empty availability, node limits,
  unknown backends, and tensor-parallel compatibility.
- Ran `git diff --check` successfully.
- No files outside the requested implementation/test scope and required report
  were changed in the worktree.

## Concerns

- Pytest emits an existing `pynvml` deprecation warning; it does not affect the
  result.
- The shared registration file contains pre-existing malformed/non-UTF-8
  historical content. This task appended records only and validated its own
  JSONL records by skipping unrelated malformed lines; no historical records
  were modified.
- Only the focused contract tests were run, as required by the task brief.

