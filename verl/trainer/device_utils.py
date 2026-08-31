"""Device selection helpers for trainer entry points."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class DeviceSelection:
    """The normalized device settings used by the Ray trainer."""

    device_name: str
    device_ids: tuple[str, ...] | None
    n_gpus_per_node: int
    nnodes: int


def _to_container(value: Any) -> Any:
    """Convert an OmegaConf list to a regular Python container when needed."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except (ImportError, TypeError, ValueError):
        return value


def _get_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return config.get(key, default)


def _set_value(config: Any, key: str, value: Any) -> None:
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
        raise RuntimeError("device: cuda requires a CUDA-enabled PyTorch installation") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("device: cuda requires at least one available CUDA GPU")
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


def configure_training_devices(
    trainer_config: MutableMapping[str, Any],
    *,
    available_device_ids: Sequence[str | int] | None = None,
    rng: Any = None,
) -> DeviceSelection:
    """Normalize ``trainer.device`` and bind the selected CUDA devices.

    ``trainer.device: cuda`` randomly selects the number of visible GPUs
    requested by ``n_gpus_per_node``. A list of integer IDs selects those
    physical GPUs and derives the single-node GPU count from the list.
    """
    device = _to_container(_get_value(trainer_config, "device", "cuda"))
    nnodes = int(_get_value(trainer_config, "nnodes", 1))
    n_gpus_per_node = int(_get_value(trainer_config, "n_gpus_per_node", 1))
    if nnodes < 1:
        raise ValueError(f"trainer.nnodes must be positive, got {nnodes}")
    if n_gpus_per_node < 1:
        raise ValueError(f"trainer.n_gpus_per_node must be positive, got {n_gpus_per_node}")

    if isinstance(device, str):
        device_name = device.strip().lower()
        if device_name != "cuda":
            return DeviceSelection(device_name, None, n_gpus_per_node, nnodes)

        if nnodes != 1:
            # Physical IDs are local to one host; Ray handles multi-node CUDA scheduling.
            return DeviceSelection("cuda", None, n_gpus_per_node, nnodes)

        source_device_ids = _visible_cuda_ids() if available_device_ids is None else available_device_ids
        candidates = [str(device_id) for device_id in source_device_ids]
        if len(candidates) < n_gpus_per_node:
            raise RuntimeError(
                f"trainer.device=cuda requested {n_gpus_per_node} GPU(s), "
                f"but only {len(candidates)} GPU(s) are available: {candidates}"
            )
        sampler = rng or random
        selected_ids = tuple(sampler.sample(candidates, n_gpus_per_node))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
        return DeviceSelection("cuda", selected_ids, n_gpus_per_node, nnodes)

    selected_ids = tuple(_normalize_gpu_ids(device))
    if nnodes != 1:
        raise ValueError("trainer.device GPU lists are only supported with trainer.nnodes=1")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
    _set_value(trainer_config, "device", "cuda")
    n_gpus_per_node = len(selected_ids)
    _set_value(trainer_config, "n_gpus_per_node", n_gpus_per_node)
    _set_value(trainer_config, "nnodes", 1)
    return DeviceSelection("cuda", selected_ids, n_gpus_per_node, 1)
