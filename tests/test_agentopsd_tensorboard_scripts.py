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
        ("run_alfworld_3b.sh", "alfworld_3b", "AgentOPSD_alfworld_turn_lambda0.5_skillfalse_3b_lora16", ["console", "tensorboard"]),
        ("run_alfworld_7b.sh", "alfworld_7b", "AgentOPSD_alfworld_turn_lambda0.5_skillfalse_7b_lora16", ["console", "wandb", "tensorboard"]),
        ("run_alfworld_7b_gpu1.sh", "alfworld_7b", "AgentOPSD_alfworld_gpu1_turn_lora16", ["console", "tensorboard"]),
        ("run_search_3b.sh", "search_3b", "AgentOPSD_search_token_lambda0.5_skillfalse_3b_lora16", ["console", "tensorboard"]),
        ("run_search_7b.sh", "search_7b", "AgentOPSD_search_token_lambda0.5_skillfalse_7b_lora16", ["console", "wandb", "tensorboard"]),
        ("run_webshop_3b.sh", "webshop_3b", "AgentOPSD_webshop_token_lambda0.5_skillfalse_3b_lora16", ["console", "tensorboard"]),
        ("run_webshop_7b.sh", "webshop_7b", "AgentOPSD_webshop_token_lambda0.5_skillfalse_7b_lora16", ["console", "wandb", "tensorboard"]),
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
    assert "actor_rollout_ref.model.lora_rank=16" in launch_args
    assert "actor_rollout_ref.model.lora_alpha=16" in launch_args
    assert "+trainer.adapter_only_checkpoint=true" in launch_args
    assert "trainer.save_freq=50" in launch_args
    assert "trainer.resume_mode=disable" in launch_args
    assert f"trainer.default_local_dir={REPO_ROOT / 'outputs' / experiment / 'checkpoints'}" in launch_args
    assert any(
        arg.startswith(f"hydra.run.dir={REPO_ROOT / 'outputs' / experiment / 'hydra'}/")
        for arg in launch_args
    )
    if "wandb" in expected_backends:
        assert f"+ray_init.runtime_env.env_vars.WANDB_DIR={REPO_ROOT / 'outputs' / experiment}" in launch_args
    if script_name.endswith("_3b.sh"):
        assert "trainer.device=[0,1]" in launch_args
        assert "local_assets.data_dir=/root/autodl-fs/datasets" in launch_args
        assert "local_assets.weights_dir=/root/autodl-fs/cache/weights" in launch_args
        assert "local_assets.model_path=/root/autodl-fs/cache/weights/Qwen2.5-3B-Instruct" in launch_args
        expected_cpus = 20 if script_name == "run_alfworld_3b.sh" else 16
        assert f"ray_init.num_cpus={expected_cpus}" in launch_args


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


def test_alfworld_3b_checkpoint_and_debug_paths_use_repo_outputs(tmp_path):
    python_stub = tmp_path / "python3"
    python_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opsd" ]; then\n'
        '    printf "SAVE_CGTD_DEBUG_DIR=%s\\n" "$SAVE_CGTD_DEBUG_DIR"\n'
        '    printf "%s\\n" "$@"\nfi\n'
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("PYTHON_BIN", None)
    env.pop("SAVE_CGTD_DEBUG_DIR", None)

    result = subprocess.run(
        ["bash", "agentopsd_trainer/run_alfworld_3b.sh", "vllm"],
        cwd=REPO_ROOT / "examples",
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    output_dir = REPO_ROOT / "outputs/AgentOPSD_alfworld_turn_lambda0.5_skillfalse_3b_lora16"
    launch_args = result.stdout.splitlines()
    assert f"trainer.default_local_dir={output_dir / 'checkpoints'}" in launch_args
    assert "+trainer.adapter_only_checkpoint=true" in launch_args
    assert f"SAVE_CGTD_DEBUG_DIR={output_dir / 'opsd_debug'}" in launch_args
    assert f"+ray_init.runtime_env.env_vars.SAVE_CGTD_DEBUG_DIR={output_dir / 'opsd_debug'}" in launch_args
    assert "+ray_init.runtime_env.env_vars.SAVE_CGTD_DEBUG=0" in launch_args
    assert (
        "+ray_init.runtime_env.env_vars.TENSORBOARD_DIR="
        "/root/tf-logs/alfworld_3b/AgentOPSD_alfworld_turn_lambda0.5_skillfalse_3b_lora16"
    ) in launch_args


def test_alfworld_3b_limits_threads_and_validation_workers(tmp_path):
    python_stub = tmp_path / "python3"
    python_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opsd" ]; then\n'
        '    printf "OPENBLAS_NUM_THREADS=%s\\n" "$OPENBLAS_NUM_THREADS"\n'
        '    printf "OMP_NUM_THREADS=%s\\n" "$OMP_NUM_THREADS"\n'
        '    printf "MKL_NUM_THREADS=%s\\n" "$MKL_NUM_THREADS"\n'
        '    printf "%s\\n" "$@"\nfi\n'
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("PYTHON_BIN", None)
    env.update(OPENBLAS_NUM_THREADS="50", OMP_NUM_THREADS="50", MKL_NUM_THREADS="50")

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "run_alfworld_3b.sh"), "vllm"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    launch_args = result.stdout.splitlines()
    assert "OPENBLAS_NUM_THREADS=1" in launch_args
    assert "OMP_NUM_THREADS=1" in launch_args
    assert "MKL_NUM_THREADS=1" in launch_args
    assert "data.val_batch_size=16" in launch_args
    assert "ray_init.num_cpus=20" in launch_args
    assert "val_data_size=128" in (SCRIPT_DIR / "run_alfworld_3b.sh").read_text()


@pytest.mark.parametrize(("script_name", "skill_name"), [
    ("run_search_3b.sh", "search"),
    ("run_webshop_3b.sh", "webshop"),
])
def test_3b_skills_path_is_independent_of_launch_directory(tmp_path, script_name, skill_name):
    python_stub = tmp_path / "python3"
    python_stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "verl.trainer.main_opsd" ]; then\n'
        '    printf "%s\\n" "$@"\nfi\n'
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    env.pop("PYTHON_BIN", None)
    result = subprocess.run(
        ["bash", f"agentopsd_trainer/{script_name}", "vllm"],
        cwd=REPO_ROOT / "examples",
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"+algorithm.opsd.skills_dir={REPO_ROOT / 'skills' / skill_name}" in result.stdout.splitlines()
