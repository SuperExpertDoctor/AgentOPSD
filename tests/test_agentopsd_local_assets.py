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
    assert config["env"]["search"]["index_path"] == "${local_assets.search_index}"
    assert config["env"]["search"]["corpus_path"] == "${local_assets.search_corpus}"
    assert config["env"]["search"]["retriever_model_path"] == "${local_assets.retriever_model}"
    assert config["env"]["webshop"]["data_dir"] == "${local_assets.webshop_dir}"


def test_agentopsd_entrypoint_uses_agentopsd_config():
    source = (REPO_ROOT / "verl/trainer/main_opsd.py").read_text()

    assert 'config_name="agentopsd_trainer"' in source


def test_agentopsd_scripts_use_local_models_and_datasets():
    scripts = sorted(SCRIPT_DIR.glob("*.sh"))
    assert len(scripts) == 6

    for script in scripts:
        source = script.read_text()
        assert "$HOME/data" not in source
        assert "Qwen/Qwen2.5" not in source
        assert DATA_DIR in source
        assert WEIGHTS_DIR in source


def test_search_retriever_launcher_uses_local_assets():
    source = (REPO_ROOT / "examples/search/retriever/retrieval_launch.sh").read_text()

    assert "$HOME/data" not in source
    assert DATA_DIR in source
    assert WEIGHTS_DIR in source
    assert "intfloat/e5-base-v2" not in source
