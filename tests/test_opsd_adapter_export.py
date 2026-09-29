from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.multiprocessing import spawn
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.trainer.ppo.opsd_ray_trainer import OPSDRayTrainer
from verl.workers import fsdp_workers


def test_opsd_adapter_only_save_skips_resume_state(tmp_path):
    calls = []
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"adapter_only_checkpoint": True, "default_local_dir": str(tmp_path)}}),
        global_steps=50,
        actor_rollout_wg=SimpleNamespace(save_checkpoint=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    OPSDRayTrainer._save_checkpoint(trainer)

    assert calls == [((str(tmp_path / "global_step_50/actor"), None, 50), {"adapter_only": True})]
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    assert not (tmp_path / "global_step_50/data.pt").exists()


def test_actor_adapter_only_save_writes_peft_files_without_fsdp_checkpoint(tmp_path, monkeypatch):
    class FakePeftModel:
        def __init__(self):
            self.peft_config = {"default": LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16,
                target_modules=["q_proj"], bias="none",
            )}

    class FakeFSDP:
        def cuda(self):
            return self

    def no_full_checkpoint(**kwargs):
        raise AssertionError("full checkpoint must not be saved")

    monkeypatch.setattr(fsdp_workers, "PeftModel", FakePeftModel)
    monkeypatch.setattr(fsdp_workers, "FSDP", FakeFSDP)
    monkeypatch.setattr(fsdp_workers, "layered_summon_lora_params", lambda model: {"q_proj.lora_A.weight": torch.ones(2, 2)})
    monkeypatch.setattr(fsdp_workers.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_workers.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(fsdp_workers.dist, "all_gather_object", lambda results, error: results.__setitem__(0, error))

    worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_lora = True
    worker._is_offload_param = False
    worker._rank = 0
    worker.actor_module = FakePeftModel()
    worker.actor_module_fsdp = FakeFSDP()
    worker.checkpoint_manager = SimpleNamespace(save_checkpoint=no_full_checkpoint)

    worker.save_checkpoint(str(tmp_path), adapter_only=True)

    adapter_dir = tmp_path / "lora_adapter"
    assert set(Path(adapter_dir).iterdir()) == {
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_config.json",
    }
    assert torch.equal(load_file(adapter_dir / "adapter_model.safetensors")["q_proj.lora_A.weight"], torch.ones(2, 2))

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fsdp_workers, "save_file", fail_write)
    with pytest.raises(RuntimeError, match="disk full"):
        worker.save_checkpoint(str(tmp_path / "failed"), adapter_only=True)


def test_actor_adapter_only_save_rejects_non_lora_model(tmp_path):
    worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_lora = False

    with pytest.raises(ValueError, match="LoRA"):
        worker.save_checkpoint(str(tmp_path), adapter_only=True)

    assert not list(tmp_path.iterdir())


def test_plain_peft_adapter_only_save_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(fsdp_workers.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_workers.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(fsdp_workers.dist, "all_gather_object", lambda results, error: results.__setitem__(0, error))

    model_config = Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
    )
    actor = get_peft_model(
        Qwen2ForCausalLM(model_config),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16, target_modules="all-linear"),
    )
    actor.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.fill_(0.25)
    expected = {key: value.clone() for key, value in get_peft_model_state_dict(actor).items()}
    worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_lora = True
    worker._is_offload_param = False
    worker._rank = 0
    worker.actor_module = actor
    worker.actor_module_fsdp = actor
    worker.save_checkpoint(str(tmp_path), adapter_only=True)

    loaded = PeftModel.from_pretrained(Qwen2ForCausalLM(model_config), str(tmp_path / "lora_adapter"))
    actual = get_peft_model_state_dict(loaded)
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP LoRA export requires CUDA")
def test_fsdp_lora_adapter_can_be_loaded_from_export(tmp_path):
    if dist.is_initialized():
        pytest.skip("requires its own process group")

    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'process_group'}", rank=0, world_size=1)
    try:
        model_config = Qwen2Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16,
            target_modules="all-linear", bias="none",
        )
        actor = get_peft_model(Qwen2ForCausalLM(model_config), lora_config)
        actor.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.fill_(0.25)
        expected = {key: value.clone().cpu() for key, value in get_peft_model_state_dict(actor).items()}
        fsdp = FSDP(
            actor.cuda(),
            auto_wrap_policy=fsdp_workers.get_fsdp_wrap_policy(
                actor, config={"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]}, is_lora=True,
            ),
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
        )
        worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
        worker._is_actor = True
        worker._is_lora = True
        worker._is_offload_param = False
        worker._rank = 0
        worker.actor_module = actor
        worker.actor_module_fsdp = fsdp
        worker.save_checkpoint(str(tmp_path), adapter_only=True)

        loaded = PeftModel.from_pretrained(
            Qwen2ForCausalLM(model_config), str(tmp_path / "lora_adapter"),
        )
        actual = get_peft_model_state_dict(loaded)
        assert set(actual) == set(expected)
        for key in expected:
            assert torch.equal(actual[key].cpu(), expected[key]), key
    finally:
        dist.destroy_process_group()


def _check_two_gpu_export(rank, output_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{output_dir}/two_gpu_group", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model_config = Qwen2Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        )
        actor = get_peft_model(
            Qwen2ForCausalLM(model_config),
            LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16, target_modules="all-linear"),
        )
        actor.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.data.fill_(0.25)
        expected = {key: value.clone().cpu() for key, value in get_peft_model_state_dict(actor).items()}
        fsdp = FSDP(
            actor.cuda(rank),
            auto_wrap_policy=fsdp_workers.get_fsdp_wrap_policy(
                actor, config={"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]}, is_lora=True,
            ),
            use_orig_params=True,
            device_id=rank,
            sync_module_states=True,
        )
        worker = fsdp_workers.ActorRolloutRefWorker.__new__(fsdp_workers.ActorRolloutRefWorker)
        worker._is_actor = True
        worker._is_lora = True
        worker._is_offload_param = False
        worker._rank = rank
        worker.actor_module = actor
        worker.actor_module_fsdp = fsdp
        worker.save_checkpoint(output_dir, adapter_only=True)
        if rank == 0:
            loaded = PeftModel.from_pretrained(
                Qwen2ForCausalLM(model_config), str(Path(output_dir) / "lora_adapter"),
            )
            actual = get_peft_model_state_dict(loaded)
            assert set(actual) == set(expected)
            for key in expected:
                assert torch.equal(actual[key].cpu(), expected[key]), key
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_fsdp_adapter_export_round_trips(tmp_path):
    spawn(_check_two_gpu_export, args=(str(tmp_path),), nprocs=2, join=True)
