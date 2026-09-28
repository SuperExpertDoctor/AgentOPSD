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
    assert "nnodes=1" in output


def test_run_opsd_cleans_runner_and_ray_after_runner_failure(monkeypatch):
    config = OmegaConf.create(
        {
            "trainer": {"device": [5, 6], "n_gpus_per_node": 2, "nnodes": 1},
            "actor_rollout_ref": {"rollout": {"tensor_model_parallel_size": 1}},
            "local_assets": {},
            "ray_init": {},
        }
    )
    selection = DeviceSelection(("5", "6"), "cuda", ("5", "6"), 2, 1)
    events = []
    ray_state = {"initialized": False}
    ray_init_kwargs = {}
    monkeypatch.setenv("AGENTOPSD_RUN_ID", "test-run-id")

    def fake_configure(trainer_config):
        events.append("resolve")
        return selection

    def fake_validate(actual_selection, tensor_parallel_size):
        assert actual_selection == selection
        assert tensor_parallel_size == 1
        events.append("validate_tp")

    def fake_ray_init(**kwargs):
        events.append("ray_init")
        ray_init_kwargs.update(kwargs)
        ray_state["initialized"] = True

    class FakeRunner:
        class RemoteRun:
            def remote(self, config):
                events.append("runner_call")
                return object()

        def __init__(self):
            self.run = self.RemoteRun()

        @classmethod
        def remote(cls):
            events.append("runner_create")
            return cls()

    def fake_ray_get(ref):
        assert ref is not None
        events.append("ray_get")
        raise RuntimeError("training failed")

    def fake_ray_kill(runner, no_restart=True):
        assert no_restart is True
        events.append("ray_kill")

    def fake_ray_shutdown():
        events.append("ray_shutdown")
        ray_state["initialized"] = False

    monkeypatch.setattr(main_opsd, "configure_training_devices", fake_configure)
    monkeypatch.setattr(main_opsd, "validate_tensor_parallel_size", fake_validate)
    monkeypatch.setattr(main_opsd, "OPSDTaskRunner", FakeRunner)
    monkeypatch.setattr(main_opsd.ray, "is_initialized", lambda: ray_state["initialized"])
    monkeypatch.setattr(main_opsd.ray, "init", fake_ray_init)
    monkeypatch.setattr(main_opsd.ray, "get", fake_ray_get)
    monkeypatch.setattr(main_opsd.ray, "kill", fake_ray_kill)
    monkeypatch.setattr(main_opsd.ray, "shutdown", fake_ray_shutdown)

    with pytest.raises(RuntimeError, match="training failed"):
        main_opsd.run_opsd(config)

    assert events == [
        "resolve",
        "validate_tp",
        "ray_init",
        "runner_create",
        "runner_call",
        "ray_get",
        "ray_kill",
        "ray_shutdown",
    ]
    assert ray_init_kwargs["runtime_env"]["env_vars"]["AGENTOPSD_RUN_ID"] == "test-run-id"
