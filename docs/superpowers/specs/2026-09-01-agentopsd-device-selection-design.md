# AgentOPSD Device Selection Design

## Objective

Make `trainer.device` the only user-facing device-selection hyperparameter for
single-node AgentOPSD runs. The selected devices must be resolved before Ray is
initialized, and Ray, FSDP, and vLLM must all be restricted to that same set.

This change addresses the current split configuration in which
`trainer.device` selects or randomizes device IDs while
`trainer.n_gpus_per_node` independently controls the Ray resource count. That
split allowed an eight-GPU launch to include GPUs already occupied by other
jobs even when the user intended to constrain the run.

## Scope

The design applies to the AgentOPSD entry point and the seven scripts under
`examples/agentopsd_trainer/` on a single node.

Multi-node device assignment, automatic free-memory scheduling, GPU process
termination, and automatic training launch are out of scope. A list-valued
`trainer.device` combined with `trainer.nnodes != 1` remains an error.

## Public Configuration Contract

`trainer.device` accepts exactly two forms:

1. `cuda`
   - Select every CUDA device that the runtime makes accessible to the process.
   - Preserve the runtime-provided device order.
   - Fail before `ray.init()` if no CUDA device is accessible.
2. A non-empty list of unique, non-negative integer physical GPU IDs, such as
   `[1, 5]`
   - Select exactly those physical GPU IDs in the supplied order.
   - The list is authoritative and replaces any inherited
     `CUDA_VISIBLE_DEVICES` value.
   - Fail before `ray.init()` when an ID is syntactically invalid or cannot be
     exposed by the runtime.

Users do not select devices with `trainer.n_gpus_per_node` or by editing
`CUDA_VISIBLE_DEVICES`. `trainer.n_gpus_per_node` remains in the resolved Hydra
configuration only because the existing Ray resource-pool interface consumes
it. Its value is always overwritten with `len(resolved_device_ids)`.

`CUDA_VISIBLE_DEVICES` is an internal binding generated during device
resolution. It is not a second public configuration surface.

## Resolution Flow

The AgentOPSD entry point performs the following steps before `ray.init()`:

1. Read the requested value from `config.trainer.device`.
2. Resolve it to an ordered tuple of device IDs.
3. Set `CUDA_VISIBLE_DEVICES` to that tuple.
4. Normalize downstream `config.trainer.device` to `cuda`, because existing
   workers expect a backend name rather than an ID list.
5. Derive `config.trainer.n_gpus_per_node` from the tuple length.
6. Validate the resolved GPU count against rollout tensor parallelism.
7. Print one device summary containing the requested value, resolved physical
   IDs, and derived GPU count.
8. Initialize Ray. Ray workers may expose their assigned card as logical
   `cuda:0`, but they cannot see a physical GPU outside the resolved tuple.

The required tensor-parallel invariant is:

```text
resolved_gpu_count >= tensor_model_parallel_size
resolved_gpu_count % tensor_model_parallel_size == 0
```

Invalid configurations fail before Ray creates workers or allocates model
memory.

## Component Changes

### Device resolver

`verl/trainer/device_utils.py` remains the single device-normalization module.
It will:

- remove `random` and the `rng` injection point;
- resolve `cuda` to all accessible devices instead of sampling
  `n_gpus_per_node` devices;
- preserve explicit-list validation for empty, duplicate, negative, and
  non-integer IDs;
- make an explicit list override inherited visibility;
- update both `trainer.device` and the derived `trainer.n_gpus_per_node`;
- return the requested value and resolved IDs in `DeviceSelection` so logging
  does not depend on the mutated Hydra configuration.

The resolver must not inspect free VRAM or silently replace a requested ID with
another GPU. Device choice must remain deterministic and attributable to
`trainer.device`.

### AgentOPSD entry point

`verl/trainer/main_opsd.py` will continue to call the resolver before Ray
initialization. It will add tensor-parallel validation using
`config.actor_rollout_ref.rollout.tensor_model_parallel_size` and emit the
resolved-device summary.

### Hydra configuration and launch scripts

`verl/trainer/config/agentopsd_trainer.yaml` will document `trainer.device` as
the sole public selector. `trainer.n_gpus_per_node` may remain as an inherited
or explicit internal default, but comments will state that it is derived and
not user-controlled.

Each script under `examples/agentopsd_trainer/` will retain a default
`trainer.device` override and the trailing `$@`, allowing an exact list to be
provided as the final Hydra override:

```bash
bash examples/agentopsd_trainer/run_alfworld_3b.sh 'trainer.device=[1,5]'
```

The scripts will remove every `trainer.n_gpus_per_node=...` override. The
single-GPU script remains explicit with `trainer.device="[1]"`; the other
scripts retain `trainer.device=cuda` as their default.

## Error Handling

The resolver or entry point must raise a direct, actionable exception before
Ray initialization for these cases:

- `trainer.device=[]`;
- duplicate, negative, non-integer, or otherwise malformed explicit IDs;
- an explicit ID that cannot be exposed by the runtime;
- `trainer.device=cuda` with no accessible CUDA devices;
- an explicit device list with `trainer.nnodes != 1`;
- a tensor-parallel size less than one;
- fewer resolved GPUs than the tensor-parallel size;
- a resolved GPU count not divisible by the tensor-parallel size.

Messages must include the rejected value and, where relevant, the resolved GPU
count and tensor-parallel size. The code must not catch these failures and fall
back to a different device selection.

Busy GPUs are deliberately not rejected by this feature because a reliable
minimum-free-memory threshold depends on model and optimizer configuration.
Users must name appropriate devices. Existing CUDA out-of-memory errors remain
the final protection when a selected GPU lacks sufficient memory.

## Tests

CPU-only unit tests will cover device resolution without starting Ray or
allocating GPU memory:

- an explicit list binds exactly those IDs and derives the resource count;
- an explicit list overrides inherited `CUDA_VISIBLE_DEVICES`;
- `cuda` selects all injected accessible devices in stable order;
- repeated `cuda` resolutions produce the same order and prove random sampling
  is gone;
- empty, duplicate, negative, non-integer, inaccessible, and no-device cases
  fail with specific messages;
- list-valued devices reject multi-node configuration;
- tensor-parallel validation accepts compatible counts and rejects undersized
  or non-divisible counts;
- device resolution occurs before the mocked `ray.init()` call;
- every AgentOPSD script declares `trainer.device`, omits
  `trainer.n_gpus_per_node`, and retains `$@` for a final CLI override.

The focused verification commands are:

```bash
pytest -q tests/test_device_utils.py
pytest -q tests/test_agentopsd_local_assets.py
```

No training, inference evaluation, simulation, rendering, or other
resource-consuming task is part of implementation verification. Any later
training launch must independently pass the repository's registration gate.

## Acceptance Criteria

The design is complete when all of the following are true:

- the only documented device-selection hyperparameter is `trainer.device`;
- `trainer.device=[1,5]` restricts the process to exactly physical GPUs 1 and
  5 before Ray initialization;
- `trainer.device=cuda` deterministically selects all runtime-accessible CUDA
  devices;
- `trainer.n_gpus_per_node` is derived from the resolved list and cannot alter
  selection;
- AgentOPSD launch scripts contain no independent GPU-count override;
- invalid selection and tensor-parallel combinations fail before worker
  creation;
- the focused CPU-only tests pass;
- no implementation or training execution begins until the user approves the
  design and implementation-plan documents.
