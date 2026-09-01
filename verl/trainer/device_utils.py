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
