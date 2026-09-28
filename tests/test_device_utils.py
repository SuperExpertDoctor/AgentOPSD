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
