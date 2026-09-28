# AgentOPSD Device Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trainer.device` the only user-facing selector for the GPUs used by single-node AgentOPSD runs.

**Architecture:** Resolve `trainer.device` once, before `ray.init()`, into an ordered physical-GPU list and bind that list through the internal `CUDA_VISIBLE_DEVICES` environment variable. Derive the existing Ray resource count from the resolved list, validate tensor parallelism against it, and remove independent GPU-count overrides from AgentOPSD launch scripts.

**Tech Stack:** Python 3.12, Hydra/OmegaConf, Ray, PyTorch CUDA discovery, pytest, Bash launch scripts.

## Global Constraints

- `trainer.device` is the only documented and user-controlled device selector.
- `trainer.device=[1,5]` selects exactly physical GPUs 1 and 5 in that order.
- `trainer.device=cuda` selects all CUDA devices accessible to the process in stable runtime order.
- `CUDA_VISIBLE_DEVICES` and `trainer.n_gpus_per_node` are internal derived state, not alternative user interfaces.
- Device resolution and tensor-parallel validation happen before `ray.init()`.
- Explicit device lists are supported only when `trainer.nnodes=1`.
- Device resolution is deterministic and never considers free VRAM or substitutes another GPU.
- Verification is CPU-only and must not start training, fine-tuning, evaluation, simulation, rendering, or another resource-consuming algorithm task.
- A later training launch is a separate operation and must first satisfy `/media/abc_disk/admin123/lbh/训练任务AI极简登记说明_v1.2.txt` and the repository `AGENTS.md` registration gate.

## File Map

- Modify `verl/trainer/device_utils.py`: deterministic parsing, binding, derived resource count, runtime-access validation, and tensor-parallel validation.
- Modify `tests/test_device_utils.py`: focused unit coverage for every accepted and rejected `device` form.
- Modify `verl/trainer/main_opsd.py`: call both device resolution and TP validation before Ray initialization and improve the device summary.
- Create `tests/test_main_opsd_device_setup.py`: verify resolver/validator/Ray call order without starting Ray.
- Modify `verl/trainer/config/agentopsd_trainer.yaml`: document `device` as the sole selector and GPU count as derived.
- Modify all seven files in `examples/agentopsd_trainer/`: remove independent `trainer.n_gpus_per_node` overrides while retaining `trainer.device` and final `$@` overrides.
- Modify `tests/test_agentopsd_local_assets.py`: enforce the launch-script configuration contract.
- Modify `README.md`: document only `trainer.device` examples.

---

### Task 1: Make Device Resolution Deterministic and Single-Sourced

**Files:**
- Modify: `tests/test_device_utils.py`
- Modify: `verl/trainer/device_utils.py`

**Interfaces:**
- Consumes: Hydra/OmegaConf-compatible `trainer_config` containing `device`, `nnodes`, and the legacy `n_gpus_per_node` field.
- Produces: `DeviceSelection(requested_device, device_name, device_ids, n_gpus_per_node, nnodes)`.
- Produces: `configure_training_devices(trainer_config, *, available_device_ids=None) -> DeviceSelection`.
- Produces: `validate_tensor_parallel_size(selection, tensor_parallel_size) -> None`.

- [ ] **Step 1: Replace the resolver tests with the new public contract**

Replace `tests/test_device_utils.py` with:

```python
import os

import pytest
from omegaconf import OmegaConf

from verl.trainer.device_utils import configure_training_devices, validate_tensor_parallel_size


def trainer_config(device, *, n_gpus_per_node=8, nnodes=1):
    return OmegaConf.create(
        {
            "device": device,
            "n_gpus_per_node": n_gpus_per_node,
            "nnodes": nnodes,
        }
    )


def test_explicit_gpu_list_is_authoritative_and_derives_resource_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config = trainer_config([1, 5])

    selection = configure_training_devices(config, available_device_ids=[0, 1, 5])

    assert selection.requested_device == ("1", "5")
    assert selection.device_name == "cuda"
    assert selection.device_ids == ("1", "5")
    assert selection.n_gpus_per_node == 2
    assert config.device == "cuda"
    assert config.n_gpus_per_node == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,5"


def test_cuda_selects_every_accessible_gpu_in_stable_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = trainer_config("cuda", n_gpus_per_node=1)

    first = configure_training_devices(config, available_device_ids=[5, 3, 0])
    second = configure_training_devices(config, available_device_ids=[5, 3, 0])

    assert first.requested_device == "cuda"
    assert first.device_ids == ("5", "3", "0")
    assert second.device_ids == first.device_ids
    assert config.n_gpus_per_node == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,3,0"


@pytest.mark.parametrize(
    ("device", "message"),
    [
        ([], "must not be empty"),
        ([1, 1], "must be unique"),
        ([-1], "must be non-negative"),
        (["1"], "must be integers"),
    ],
)
def test_invalid_explicit_gpu_lists_are_rejected(device, message):
    with pytest.raises(ValueError, match=message):
        configure_training_devices(trainer_config(device), available_device_ids=[0, 1])


def test_inaccessible_explicit_gpu_is_rejected():
    with pytest.raises(RuntimeError, match=r"requested GPU IDs.*5.*accessible.*0.*1"):
        configure_training_devices(trainer_config([1, 5]), available_device_ids=[0, 1])


def test_cuda_rejects_empty_accessible_device_set(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="no accessible CUDA GPUs"):
        configure_training_devices(trainer_config("cuda"), available_device_ids=[])


def test_explicit_gpu_list_requires_single_node():
    with pytest.raises(ValueError, match="nnodes=1"):
        configure_training_devices(trainer_config([1, 2], nnodes=2), available_device_ids=[0, 1, 2])


def test_unknown_device_backend_is_rejected():
    with pytest.raises(ValueError, match="must be 'cuda' or a list"):
        configure_training_devices(trainer_config("cpu"), available_device_ids=[0, 1])


@pytest.mark.parametrize(("gpu_count", "tp_size"), [(2, 1), (2, 2), (8, 2)])
def test_tensor_parallel_size_accepts_compatible_gpu_counts(gpu_count, tp_size):
    selection = configure_training_devices(
        trainer_config(list(range(gpu_count))),
        available_device_ids=list(range(gpu_count)),
    )

    validate_tensor_parallel_size(selection, tp_size)


@pytest.mark.parametrize(
    ("gpu_count", "tp_size", "message"),
    [
        (2, 0, "must be positive"),
        (1, 2, "resolved GPU count 1 is smaller"),
        (3, 2, "resolved GPU count 3 must be divisible"),
    ],
)
def test_tensor_parallel_size_rejects_incompatible_gpu_counts(gpu_count, tp_size, message):
    selection = configure_training_devices(
        trainer_config(list(range(gpu_count))),
        available_device_ids=list(range(gpu_count)),
    )

    with pytest.raises(ValueError, match=message):
        validate_tensor_parallel_size(selection, tp_size)
```

- [ ] **Step 2: Run the new tests and confirm they fail against the random/count-driven resolver**

Run:

```bash
pytest -q tests/test_device_utils.py
```

Expected: FAIL during collection because `validate_tensor_parallel_size` does
not exist, or fail assertions showing that `cuda` samples only the legacy
`n_gpus_per_node` count and inaccessible explicit IDs are not rejected.

- [ ] **Step 3: Replace the resolver with deterministic `device` semantics**

Replace `verl/trainer/device_utils.py` with:

```python
"""Device selection helpers for AgentOPSD trainer entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class DeviceSelection:
    """Requested and normalized device settings used by the Ray trainer."""

    requested_device: str | tuple[str, ...]
    device_name: str
    device_ids: tuple[str, ...] | None
    n_gpus_per_node: int
    nnodes: int


def _to_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except (ImportError, TypeError, ValueError):
        return value


def _get_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return config.get(key, default)


def _set_value(config: MutableMapping[str, Any], key: str, value: Any) -> None:
    config[key] = value


def _visible_cuda_ids() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        if visible_devices in {"", "NoDevFiles"}:
            return []
        return [device.strip() for device in visible_devices.split(",") if device.strip()]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("trainer.device=cuda requires a CUDA-enabled PyTorch installation") from exc

    if not torch.cuda.is_available():
        return []
    return [str(device_id) for device_id in range(torch.cuda.device_count())]


def _normalize_gpu_ids(value: Any) -> list[str]:
    value = _to_container(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("trainer.device must be 'cuda' or a list of GPU integer IDs")
    if not value:
        raise ValueError("trainer.device GPU list must not be empty")

    normalized_ids = []
    for device_id in value:
        if isinstance(device_id, bool) or not isinstance(device_id, Integral):
            raise ValueError(f"trainer.device GPU IDs must be integers, got {device_id!r}")
        if device_id < 0:
            raise ValueError(f"trainer.device GPU IDs must be non-negative, got {device_id}")
        normalized_ids.append(str(int(device_id)))

    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"trainer.device GPU IDs must be unique, got {value!r}")
    return normalized_ids


def _validate_injected_access(selected_ids: Sequence[str], accessible_ids: Sequence[str]) -> None:
    accessible_set = set(accessible_ids)
    inaccessible_ids = [device_id for device_id in selected_ids if device_id not in accessible_set]
    if inaccessible_ids:
        raise RuntimeError(
            f"trainer.device requested GPU IDs {list(selected_ids)}, but IDs {inaccessible_ids} "
            f"are not accessible; accessible GPU IDs are {list(accessible_ids)}"
        )


def _validate_runtime_binding(selected_ids: Sequence[str]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("trainer.device requires a CUDA-enabled PyTorch installation") from exc

    visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_count != len(selected_ids):
        raise RuntimeError(
            f"trainer.device requested GPU IDs {list(selected_ids)}, but the runtime exposed "
            f"{visible_count} CUDA GPU(s) after binding"
        )


def configure_training_devices(
    trainer_config: MutableMapping[str, Any],
    *,
    available_device_ids: Sequence[str | int] | None = None,
) -> DeviceSelection:
    """Resolve ``trainer.device`` and bind the selected CUDA devices."""

    device = _to_container(_get_value(trainer_config, "device", "cuda"))
    nnodes = int(_get_value(trainer_config, "nnodes", 1))
    if nnodes < 1:
        raise ValueError(f"trainer.nnodes must be positive, got {nnodes}")

    injected_ids = None
    if available_device_ids is not None:
        injected_ids = [str(device_id) for device_id in available_device_ids]

    if isinstance(device, str):
        device_name = device.strip().lower()
        if device_name != "cuda":
            raise ValueError("trainer.device must be 'cuda' or a list of GPU integer IDs")
        if nnodes != 1:
            n_gpus_per_node = int(_get_value(trainer_config, "n_gpus_per_node", 1))
            return DeviceSelection("cuda", "cuda", None, n_gpus_per_node, nnodes)

        selected_ids = injected_ids if injected_ids is not None else _visible_cuda_ids()
        if not selected_ids:
            raise RuntimeError("trainer.device=cuda found no accessible CUDA GPUs")
        requested_device: str | tuple[str, ...] = "cuda"
    else:
        if nnodes != 1:
            raise ValueError("trainer.device GPU lists are only supported with trainer.nnodes=1")
        selected_ids = _normalize_gpu_ids(device)
        requested_device = tuple(selected_ids)
        if injected_ids is not None:
            _validate_injected_access(selected_ids, injected_ids)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
    if injected_ids is None:
        _validate_runtime_binding(selected_ids)

    n_gpus_per_node = len(selected_ids)
    _set_value(trainer_config, "device", "cuda")
    _set_value(trainer_config, "n_gpus_per_node", n_gpus_per_node)
    _set_value(trainer_config, "nnodes", 1)
    return DeviceSelection(requested_device, "cuda", tuple(selected_ids), n_gpus_per_node, 1)


def validate_tensor_parallel_size(selection: DeviceSelection, tensor_parallel_size: Any) -> None:
    """Fail before Ray starts when the selected GPUs cannot form TP groups."""

    tp_size = int(tensor_parallel_size)
    if tp_size < 1:
        raise ValueError(f"tensor_model_parallel_size must be positive, got {tp_size}")

    gpu_count = selection.n_gpus_per_node
    if gpu_count < tp_size:
        raise ValueError(
            f"resolved GPU count {gpu_count} is smaller than tensor_model_parallel_size {tp_size}"
        )
    if gpu_count % tp_size != 0:
        raise ValueError(
            f"resolved GPU count {gpu_count} must be divisible by tensor_model_parallel_size {tp_size}"
        )
```

- [ ] **Step 4: Run the focused resolver tests**

Run:

```bash
pytest -q tests/test_device_utils.py
```

Expected: all tests PASS without Ray startup or CUDA allocation. Tests pass an
injected device inventory, so they remain CPU-only.

- [ ] **Step 5: Commit the deterministic resolver**

```bash
git add verl/trainer/device_utils.py tests/test_device_utils.py
git commit -m "fix: make AgentOPSD device selection deterministic"
```

---

### Task 2: Validate Devices Before Ray Initialization

**Files:**
- Create: `tests/test_main_opsd_device_setup.py`
- Modify: `verl/trainer/main_opsd.py:12-29`

**Interfaces:**
- Consumes: `configure_training_devices(config.trainer) -> DeviceSelection` from Task 1.
- Consumes: `validate_tensor_parallel_size(selection, tensor_parallel_size) -> None` from Task 1.
- Produces: a pre-Ray device summary with requested selection, resolved physical IDs, and derived count.

- [ ] **Step 1: Add a test that stops at `ray.init()` and records call order**

Create `tests/test_main_opsd_device_setup.py`:

```python
import pytest
from omegaconf import OmegaConf

from verl.trainer import main_opsd
from verl.trainer.device_utils import DeviceSelection


class StopAtRayInit(Exception):
    pass


def test_device_resolution_and_tp_validation_run_before_ray_init(monkeypatch, capsys):
    config = OmegaConf.create(
        {
            "trainer": {"device": [1, 5], "n_gpus_per_node": 8, "nnodes": 1},
            "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 2}},
            "local_assets": {},
            "ray_init": {},
        }
    )
    selection = DeviceSelection(("1", "5"), "cuda", ("1", "5"), 2, 1)
    events = []

    def fake_configure(trainer_config):
        events.append("resolve")
        trainer_config.device = "cuda"
        trainer_config.n_gpus_per_node = 2
        return selection

    def fake_validate(actual_selection, tensor_parallel_size):
        assert actual_selection == selection
        assert tensor_parallel_size == 2
        events.append("validate_tp")

    def fake_ray_init(**kwargs):
        events.append("ray_init")
        raise StopAtRayInit

    monkeypatch.setattr(main_opsd, "configure_training_devices", fake_configure)
    monkeypatch.setattr(main_opsd, "validate_tensor_parallel_size", fake_validate)
    monkeypatch.setattr(main_opsd.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_opsd.ray, "init", fake_ray_init)

    with pytest.raises(StopAtRayInit):
        main_opsd.run_opsd(config)

    assert events == ["resolve", "validate_tp", "ray_init"]
    output = capsys.readouterr().out
    assert "requested=('1', '5')" in output
    assert "resolved_physical_gpu_ids=('1', '5')" in output
    assert "n_gpus_per_node=2" in output
```

- [ ] **Step 2: Run the entry-point test and confirm it fails**

Run:

```bash
pytest -q tests/test_main_opsd_device_setup.py
```

Expected: FAIL because `main_opsd` does not import or call
`validate_tensor_parallel_size`, and the existing log omits the requested
selection.

- [ ] **Step 3: Wire TP validation and the expanded summary into the entry point**

Change the import in `verl/trainer/main_opsd.py` to:

```python
from verl.trainer.device_utils import configure_training_devices, validate_tensor_parallel_size
```

Replace the opening device block in `run_opsd` with:

```python
def run_opsd(config) -> None:
    device_selection = configure_training_devices(config.trainer)
    validate_tensor_parallel_size(
        device_selection,
        config.actor_rollout_ref.rollout.tensor_model_parallel_size,
    )
    print(
        "[device] backend={} requested={} resolved_physical_gpu_ids={} "
        "n_gpus_per_node={} nnodes={}".format(
            device_selection.device_name,
            device_selection.requested_device,
            device_selection.device_ids,
            device_selection.n_gpus_per_node,
            device_selection.nnodes,
        )
    )
```

Keep the existing `local_assets` handling and `ray.init()` block immediately
after this replacement. Do not move CUDA-dependent imports ahead of device
resolution.

- [ ] **Step 4: Run the entry-point and resolver tests**

Run:

```bash
pytest -q tests/test_main_opsd_device_setup.py tests/test_device_utils.py
```

Expected: all tests PASS. The sentinel prevents `ray.init()` from creating a
Ray runtime.

- [ ] **Step 5: Commit the pre-Ray validation**

```bash
git add verl/trainer/main_opsd.py tests/test_main_opsd_device_setup.py
git commit -m "fix: validate AgentOPSD devices before Ray startup"
```

---

### Task 3: Remove Independent GPU Counts and Document `device`

**Files:**
- Modify: `verl/trainer/config/agentopsd_trainer.yaml:5-9`
- Modify: `examples/agentopsd_trainer/run_alfworld_3b.sh`
- Modify: `examples/agentopsd_trainer/run_alfworld_7b.sh`
- Modify: `examples/agentopsd_trainer/run_alfworld_7b_gpu1.sh`
- Modify: `examples/agentopsd_trainer/run_search_3b.sh`
- Modify: `examples/agentopsd_trainer/run_search_7b.sh`
- Modify: `examples/agentopsd_trainer/run_webshop_3b.sh`
- Modify: `examples/agentopsd_trainer/run_webshop_7b.sh`
- Modify: `tests/test_agentopsd_local_assets.py:35-63`
- Modify: `README.md:128-142`

**Interfaces:**
- Consumes: the Task 1 contract that derives `n_gpus_per_node` from `device`.
- Produces: launch scripts in which the final Hydra `trainer.device` value is the only user device selection.
- Produces: user documentation for `trainer.device=cuda` and `trainer.device=[...]`.

- [ ] **Step 1: Strengthen configuration and script contract tests**

In `tests/test_agentopsd_local_assets.py`, replace
`test_agentopsd_config_uses_explicit_gpu_device_list` with:

```python
def test_agentopsd_config_declares_device_as_the_device_selector():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["trainer"]["device"] == [1]
    assert config["trainer"]["n_gpus_per_node"] == 1
    assert config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.85
```

Replace the device assertions inside
`test_agentopsd_scripts_use_local_models_and_datasets` with:

```python
        if script.name == "run_alfworld_7b_gpu1.sh":
            assert 'trainer.device="[1]"' in source
        else:
            assert "trainer.device=cuda" in source
        assert "trainer.n_gpus_per_node" not in source
        assert "$@" in source
        assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.85" in source
```

- [ ] **Step 2: Run the static contract test and confirm it fails**

Run:

```bash
pytest -q tests/test_agentopsd_local_assets.py
```

Expected: FAIL because the AgentOPSD scripts still contain independent
`trainer.n_gpus_per_node` overrides.

- [ ] **Step 3: Clarify the Hydra configuration contract**

Replace the `trainer` block at the top of
`verl/trainer/config/agentopsd_trainer.yaml` with:

```yaml
trainer:
  # Sole user-facing device selector: cuda uses all accessible GPUs; a list
  # such as [1, 5] binds exactly those physical GPU IDs.
  device: [1]
  # Internal compatibility field; configure_training_devices derives it from device.
  n_gpus_per_node: 1
  nnodes: 1
```

- [ ] **Step 4: Remove every independent GPU-count override from AgentOPSD scripts**

Apply these exact deletions and make no other script changes:

```diff
*** Update File: examples/agentopsd_trainer/run_alfworld_3b.sh
@@
-    trainer.n_gpus_per_node=8 \
*** Update File: examples/agentopsd_trainer/run_alfworld_7b.sh
@@
-    trainer.n_gpus_per_node=8 \
*** Update File: examples/agentopsd_trainer/run_alfworld_7b_gpu1.sh
@@
-    trainer.n_gpus_per_node=1 \
*** Update File: examples/agentopsd_trainer/run_search_3b.sh
@@
-    trainer.n_gpus_per_node=4 \
*** Update File: examples/agentopsd_trainer/run_search_7b.sh
@@
-    trainer.n_gpus_per_node=4 \
*** Update File: examples/agentopsd_trainer/run_webshop_3b.sh
@@
-    trainer.n_gpus_per_node=2 \
*** Update File: examples/agentopsd_trainer/run_webshop_7b.sh
@@
-    trainer.n_gpus_per_node=2 \
```

- [ ] **Step 5: Document `trainer.device` as the only selection interface**

Insert the following after the AgentOPSD command block in `README.md`:

````markdown
Select training GPUs only with the `trainer.device` hyperparameter. Pass an
explicit list to bind exact physical GPU IDs:

```bash
bash examples/agentopsd_trainer/run_alfworld_3b.sh 'trainer.device=[1,5]'
```

Use `trainer.device=cuda` to select every CUDA GPU accessible to the process.
The program derives the Ray GPU count from `trainer.device`; do not pass
`trainer.n_gpus_per_node` for AgentOPSD runs.
````

Do not document `CUDA_VISIBLE_DEVICES` as a user selection mechanism.

- [ ] **Step 6: Run the configuration and script tests**

Run:

```bash
pytest -q tests/test_agentopsd_local_assets.py
```

Expected: all tests PASS.

- [ ] **Step 7: Run the complete CPU-only verification set**

Run:

```bash
pytest -q \
  tests/test_device_utils.py \
  tests/test_main_opsd_device_setup.py \
  tests/test_agentopsd_local_assets.py
git diff --check
```

Expected: all tests PASS and `git diff --check` prints no errors. Do not run an
AgentOPSD shell script as part of this verification.

- [ ] **Step 8: Commit scripts, configuration, and documentation**

```bash
git add \
  README.md \
  verl/trainer/config/agentopsd_trainer.yaml \
  examples/agentopsd_trainer \
  tests/test_agentopsd_local_assets.py
git commit -m "docs: make device the AgentOPSD GPU selector"
```

---

## Final Review Gate

After all three task commits exist, inspect the final diff and rerun only the
CPU-only verification commands from Task 3. Do not launch training. Present the
implementation diff and test output to the user for approval before any
resource-consuming validation or training run.
