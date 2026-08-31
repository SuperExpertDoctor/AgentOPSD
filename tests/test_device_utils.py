import os

import pytest
from omegaconf import OmegaConf

from verl.trainer.device_utils import configure_training_devices


def test_explicit_gpu_list_binds_devices_and_updates_resource_count(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = OmegaConf.create({"device": [1, 2], "n_gpus_per_node": 8, "nnodes": 1})

    selection = configure_training_devices(config)

    assert selection.device_name == "cuda"
    assert selection.device_ids == ("1", "2")
    assert config.device == "cuda"
    assert config.n_gpus_per_node == 2
    assert config.nnodes == 1
    assert selection.nnodes == 1
    assert selection.n_gpus_per_node == 2
    assert selection.device_ids == tuple(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def test_cuda_randomly_selects_requested_number_of_visible_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = OmegaConf.create({"device": "cuda", "n_gpus_per_node": 2, "nnodes": 1})

    class FixedSampler:
        def sample(self, population, count):
            assert population == ["0", "1", "2"]
            assert count == 2
            return ["2", "0"]

    selection = configure_training_devices(
        config,
        available_device_ids=[0, 1, 2],
        rng=FixedSampler(),
    )

    assert selection.device_ids == ("2", "0")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,0"


def test_empty_gpu_list_is_rejected():
    config = OmegaConf.create({"device": [], "n_gpus_per_node": 1, "nnodes": 1})

    with pytest.raises(ValueError, match="must not be empty"):
        configure_training_devices(config)


def test_gpu_list_requires_single_node():
    config = OmegaConf.create({"device": [1, 2], "n_gpus_per_node": 2, "nnodes": 2})

    with pytest.raises(ValueError, match="nnodes=1"):
        configure_training_devices(config)


def test_cuda_rejects_empty_available_gpu_set(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    config = OmegaConf.create({"device": "cuda", "n_gpus_per_node": 1, "nnodes": 1})

    with pytest.raises(RuntimeError, match=r"only 0 GPU\(s\) are available"):
        configure_training_devices(config, available_device_ids=[])
