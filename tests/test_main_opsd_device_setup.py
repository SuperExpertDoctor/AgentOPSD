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
