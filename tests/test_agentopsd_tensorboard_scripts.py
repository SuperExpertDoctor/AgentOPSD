import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "examples/agentopsd_trainer"


@pytest.mark.parametrize(
    ("script_name", "log_group", "experiment", "expected_backends"),
    [
        ("run_alfworld_3b.sh", "alfworld_3b", "AgentOPSD_alfworld_turn_lambda0.5_skillfalse", ["console", "tensorboard"]),
        ("run_alfworld_7b.sh", "alfworld_7b", "AgentOPSD_alfworld_turn_lambda0.5_skillfalse", ["console", "wandb", "tensorboard"]),
        ("run_alfworld_7b_gpu1.sh", "alfworld_7b", "AgentOPSD_alfworld_gpu1_turn_lora16", ["console", "tensorboard"]),
        ("run_search_3b.sh", "search_3b", "AgentOPSD_search_token_lambda0.5_skillfalse", ["console", "tensorboard"]),
        ("run_search_7b.sh", "search_7b", "AgentOPSD_search_token_lambda0.5_skillfalse", ["console", "wandb", "tensorboard"]),
        ("run_webshop_3b.sh", "webshop_3b", "AgentOPSD_webshop_token_lambda0.5_skillfalse", ["console", "tensorboard"]),
        ("run_webshop_7b.sh", "webshop_7b", "AgentOPSD_webshop_token_lambda0.5_skillfalse", ["console", "wandb", "tensorboard"]),
    ],
)
def test_launch_scripts_pass_tensorboard_backend_and_log_dir(
    tmp_path, script_name, log_group, experiment, expected_backends
):
    # Capture the launch arguments without running preprocessing or training.
    python_stub = tmp_path / "python3"
    python_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opsd" ]; then\n'
        '    printf "%s\\n" "$@"\nfi\n'
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("PYTHON_BIN", None)
    script = SCRIPT_DIR / script_name
    args = ["bash", str(script)]
    if script_name != "run_alfworld_7b_gpu1.sh":
        args.append("vllm")

    result = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    launch_args = result.stdout.splitlines()
    assert launch_args[:2] == ["-m", "verl.trainer.main_opsd"]
    logger_arg = next(arg for arg in launch_args if arg.startswith("trainer.logger="))
    assert yaml.safe_load(logger_arg.partition("=")[2]) == expected_backends
    assert (
        f"+ray_init.runtime_env.env_vars.TENSORBOARD_DIR=/root/tf-logs/{log_group}/{experiment}"
        in launch_args
    )
    if script_name.endswith("_3b.sh"):
        assert "trainer.device=[0,1]" in launch_args
        assert "local_assets.data_dir=/root/autodl-fs/datasets" in launch_args
        assert "local_assets.weights_dir=/root/autodl-fs/cache/weights" in launch_args
        assert "local_assets.model_path=/root/autodl-fs/cache/weights/Qwen2.5-3B-Instruct" in launch_args
        assert "ray_init.num_cpus=16" in launch_args


def test_alfworld_3b_skills_path_is_independent_of_launch_directory(tmp_path):
    python_stub = tmp_path / "python3"
    python_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opsd" ]; then\n'
        '    printf "%s\\n" "$@"\nfi\n'
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("PYTHON_BIN", None)

    result = subprocess.run(
        ["bash", "agentopsd_trainer/run_alfworld_3b.sh"],
        cwd=REPO_ROOT / "examples",
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"+algorithm.opsd.skills_dir={REPO_ROOT / 'skills/alfworld'}" in result.stdout.splitlines()
