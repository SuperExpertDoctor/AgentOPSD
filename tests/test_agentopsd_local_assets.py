from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "verl/trainer/config/agentopsd_trainer.yaml"
SCRIPT_DIR = REPO_ROOT / "examples/agentopsd_trainer"
DATA_DIR = "/home/shuixia/users/houguoqiang/code/datasets"
WEIGHTS_DIR = "/home/shuixia/users/houguoqiang/code/weights"


def test_agentopsd_hydra_config_resolves_local_asset_paths():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["local_assets"]["data_dir"] == DATA_DIR
    assert config["local_assets"]["weights_dir"] == WEIGHTS_DIR
    assert config["data"]["train_files"] == "${local_assets.text_train_file}"
    assert config["data"]["val_files"] == "${local_assets.text_val_file}"
    assert config["actor_rollout_ref"]["model"]["path"] == "${local_assets.model_path}"
    assert config["critic"]["model"]["path"] == "${actor_rollout_ref.model.path}"
    assert config["env"]["alfworld"]["data_dir"] == "${local_assets.alfworld_dir}"
    assert config["env"]["alfworld"]["actor_startup_batch_size"] == 16
    assert config["env"]["search"]["index_path"] == "${local_assets.search_index}"
    assert config["env"]["search"]["corpus_path"] == "${local_assets.search_corpus}"
    assert config["env"]["search"]["retriever_model_path"] == "${local_assets.retriever_model}"
    assert config["env"]["webshop"]["data_dir"] == "${local_assets.webshop_dir}"


def test_agentopsd_entrypoint_uses_agentopsd_config():
    source = (REPO_ROOT / "verl/trainer/main_opsd.py").read_text()

    assert 'config_name="agentopsd_trainer"' in source


def test_agentopsd_config_declares_device_as_the_device_selector():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["trainer"]["device"] == [1]
    assert config["trainer"]["n_gpus_per_node"] == 1
    assert config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] == 0.5
    assert config["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] == 4096
    assert config["actor_rollout_ref"]["rollout"]["max_num_seqs"] == 128
    assert config["actor_rollout_ref"]["rollout"]["enforce_eager"] is True


def test_agentopsd_entrypoint_exposes_device_configuration():
    from verl.trainer import main_opsd

    assert hasattr(main_opsd, "configure_training_devices")


def test_agentopsd_initializes_training_workers_before_environment_actors():
    source = (REPO_ROOT / "verl/trainer/main_opsd.py").read_text()

    assert source.index("trainer.init_workers()") < source.index("envs, val_envs = make_envs(config)")
    assert "trainer.envs = envs" in source
    assert "trainer.val_envs = val_envs" in source


def test_agentopsd_main_has_exception_cleanup_and_fsdp_startup_guard():
    main_source = (REPO_ROOT / "verl/trainer/main_opsd.py").read_text()
    fsdp_source = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()

    assert "finally:" in main_source
    assert "cleanup_runtime_resources" in main_source
    assert "ray.shutdown()" in main_source
    assert "sync_module_states = not bool(" in fsdp_source
    assert 'getattr(actor_model_config, "tie_word_embeddings", False)' in fsdp_source
    assert "sync_module_states=sync_module_states" in fsdp_source


def test_alfworld_workers_are_created_in_ready_batches(monkeypatch):
    alfworld_root = REPO_ROOT / "agent_system/environments/env_package/alfworld"
    monkeypatch.syspath_prepend(str(alfworld_root))
    from agent_system.environments.env_package.alfworld import envs as alfworld_envs

    events = []

    class RemoteMethod:
        def __init__(self, worker_id):
            self.worker_id = worker_id

        def remote(self):
            events.append(("ready", self.worker_id))
            return self.worker_id

    class FakeWorker:
        def __init__(self, worker_id):
            self.worker_id = worker_id
            self.ready = RemoteMethod(worker_id)

    class FakeWorkerClass:
        def remote(self, config, seed, base_env):
            worker_id = len([event for event in events if event[0] == "create"])
            events.append(("create", worker_id, seed))
            return FakeWorker(worker_id)

    def fake_get(refs):
        events.append(("barrier", tuple(refs)))
        return refs

    monkeypatch.setattr(alfworld_envs.ray, "get", fake_get)
    workers = alfworld_envs._create_ready_workers(
        FakeWorkerClass(),
        config={},
        seed=10,
        base_env=object(),
        num_processes=5,
        group_n=2,
        startup_batch_size=2,
    )

    assert len(workers) == 5
    assert events == [
        ("create", 0, 10),
        ("create", 1, 10),
        ("ready", 0),
        ("ready", 1),
        ("barrier", (0, 1)),
        ("create", 2, 11),
        ("create", 3, 11),
        ("ready", 2),
        ("ready", 3),
        ("barrier", (2, 3)),
        ("create", 4, 12),
        ("ready", 4),
        ("barrier", (4,)),
    ]


def test_agentopsd_scripts_use_local_models_and_datasets():
    scripts = sorted(SCRIPT_DIR.glob("*.sh"))
    assert len(scripts) == 7

    for script in scripts:
        source = script.read_text()
        assert "$HOME/data" not in source
        assert "Qwen/Qwen2.5" not in source
        assert DATA_DIR in source
        assert WEIGHTS_DIR in source
        assert "../process_cleanup/agentopsd_process_cleanup.sh" in source
        assert "agentopsd_cleanup_setup" in source
        assert "verl.trainer.agentopsd_cleanup preflight" not in source
        assert "flock" not in source
        if script.name == "run_alfworld_7b_gpu1.sh":
            assert 'trainer.device="[1]"' in source
        elif script.name == "run_alfworld_3b.sh":
            assert 'PYTHON_BIN="${PYTHON_BIN:-/home/shuixia/miniconda3/envs/agentopsd/bin/python}"' in source
            assert 'trainer.device="[2,3]"' in source
            assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in source
            assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8" in source
            assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in source
            assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.5" in source
            assert "actor_rollout_ref.rollout.max_num_batched_tokens=4096" in source
            assert "actor_rollout_ref.rollout.max_num_seqs=128" in source
            assert "actor_rollout_ref.rollout.enforce_eager=True" in source
            assert "PYTORCH_CUDA_ALLOC_CONF" not in source
            assert "trainer.logger=['console']" in source
            assert "WANDB_API_KEY=your_key_here" not in source
        else:
            assert "trainer.device=cuda" in source
        assert "trainer.n_gpus_per_node" not in source
        assert "$@" in source
        if script.name != "run_alfworld_3b.sh":
            assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.85" in source


def test_search_retriever_launcher_uses_local_assets():
    source = (REPO_ROOT / "examples/search/retriever/retrieval_launch.sh").read_text()

    assert "$HOME/data" not in source
    assert DATA_DIR in source
    assert WEIGHTS_DIR in source
    assert "intfloat/e5-base-v2" not in source
