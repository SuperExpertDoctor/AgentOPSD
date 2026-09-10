"""Best-effort teardown for resources owned by one AgentOPSD trainer run."""

from __future__ import annotations

import gc
from typing import Iterable, Sequence

try:
    from ray.util.placement_group import remove_placement_group
except ImportError:  # pragma: no cover - Ray is present in the training env.
    remove_placement_group = None


RUN_ID_ENV = "AGENTOPSD_RUN_ID"


def _actor_key(actor_handle) -> tuple[str, str]:
    actor_id = getattr(actor_handle, "_actor_id", None)
    if actor_id is not None:
        try:
            actor_id = actor_id.hex()
        except AttributeError:
            actor_id = str(actor_id)
        return "actor", actor_id
    return "object", str(id(actor_handle))


def _iter_worker_handles(trainer) -> Iterable[object]:
    seen = set()
    groups = [
        getattr(trainer, group_name, None)
        for group_name in ("actor_rollout_wg", "critic_wg", "ref_policy_wg", "rm_wg")
    ]
    groups.extend(getattr(trainer, "_worker_groups", None) or [])
    for group in groups:
        if group is None:
            continue
        group_items = group.values() if isinstance(group, dict) else (group,)
        for group_item in group_items:
            try:
                workers = group_item.workers
            except Exception:
                continue
            for worker in workers or []:
                key = _actor_key(worker)
                if key in seen:
                    continue
                seen.add(key)
                yield worker


def _iter_environment_objects(environments) -> Iterable[object]:
    if environments is None:
        return
    if isinstance(environments, (list, tuple, set)):
        items = environments
    else:
        items = (environments,)
    seen = set()
    for item in items:
        if item is None:
            continue
        environment = getattr(item, "envs", item)
        key = id(environment)
        if key in seen:
            continue
        seen.add(key)
        yield environment


def cleanup_runtime_resources(
    trainer=None,
    envs: Sequence[object] | object = (),
    *,
    ray_module=None,
) -> tuple[str, ...]:
    """Release resources owned by one trainer invocation without stopping Ray."""

    errors = []

    for environment in _iter_environment_objects(envs):
        close = getattr(environment, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as exc:  # pragma: no cover - depends on Ray state.
            errors.append(f"environment close: {exc}")

    if ray_module is None:
        try:
            import ray as ray_module
        except ImportError:
            ray_module = None

    if ray_module is not None and trainer is not None:
        for worker in _iter_worker_handles(trainer):
            try:
                ray_module.kill(worker, no_restart=True)
            except TypeError:
                try:
                    ray_module.kill(worker)
                except Exception as exc:  # pragma: no cover
                    errors.append(f"worker kill: {exc}")
            except Exception as exc:  # pragma: no cover - depends on Ray state.
                errors.append(f"worker kill: {exc}")

    if trainer is not None and remove_placement_group is not None:
        resource_pool_manager = getattr(trainer, "resource_pool_manager", None)
        pools = getattr(resource_pool_manager, "resource_pool_dict", {})
        for pool in pools.values():
            for placement_group in getattr(pool, "pgs", None) or []:
                try:
                    remove_placement_group(placement_group)
                except Exception as exc:  # pragma: no cover - depends on Ray state.
                    errors.append(f"placement group removal: {exc}")

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - host dependent.
        errors.append(f"CUDA cache cleanup: {exc}")

    return tuple(errors)
