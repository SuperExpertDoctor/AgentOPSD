import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "download" / "download_agentopsd_assets.py"


def load_downloader():
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_agentopsd_assets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_downloader_exposes_help_without_downloading():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout
    assert "--data-dir" in result.stdout
    assert "--weights-dir" in result.stdout


def test_default_output_directories_are_separate_and_fixed():
    downloader = load_downloader()

    assert downloader.DEFAULT_DATA_DIR == Path(
        "/home/shuixia/users/houguoqiang/code/datasets"
    )
    assert downloader.DEFAULT_WEIGHTS_DIR == Path(
        "/home/shuixia/users/houguoqiang/code/weights"
    )


def test_all_manifest_covers_required_models_and_data_sources():
    downloader = load_downloader()
    entries = downloader.build_manifest(["all"])

    names = {entry.name for entry in entries}
    sources = {entry.source for entry in entries}

    assert "Qwen/Qwen2.5-3B-Instruct" in names
    assert "Qwen/Qwen2.5-7B-Instruct" in names
    assert "intfloat/e5-base-v2" in names
    assert "hiyouga/geometry3k" in names
    assert "hf://datasets/PeterJinGo/wiki-18-e5-index" in sources
    assert "hf://datasets/PeterJinGo/wiki-18-corpus" in sources
    assert "hf://datasets/HongbangYuan/webshop/items_shuffle.json" in sources
    assert any(entry.name == "items_shuffle.json" for entry in entries)
    assert any(entry.name == "mrcnn_alfred_objects_sep13_004.pth" for entry in entries)

    for entry in entries:
        assert not Path(entry.relative_path).is_absolute()

    assert all(
        entry.kind == "weight"
        for entry in entries
        if entry.task == "models" or entry.name == "mrcnn_alfred_objects_sep13_004.pth"
    )


def test_task_selection_and_webshop_size_are_respected():
    downloader = load_downloader()
    entries = downloader.build_manifest(["webshop"], webshop_size="small")

    assert {entry.name for entry in entries} == {
        "items_shuffle_1000.json",
        "items_ins_v2_1000.json",
        "items_human_ins.json",
    }
    assert downloader.normalize_tasks("search") == ("search",)


def test_dry_run_prints_plan_without_creating_output_directories(tmp_path):
    data_dir = tmp_path / "datasets"
    weights_dir = tmp_path / "weights"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "models",
            "alfworld",
            "--data-dir",
            str(data_dir),
            "--weights-dir",
            str(weights_dir),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(data_dir) in result.stdout
    assert str(weights_dir) in result.stdout
    assert "Qwen/Qwen2.5-3B-Instruct" in result.stdout
    assert "json_2.1.1_json.zip" in result.stdout
    assert not data_dir.exists()
    assert not weights_dir.exists()


def test_search_row_is_converted_to_tool_aware_training_record():
    downloader = load_downloader()

    class Row(dict):
        name = 17

    row = Row(
        question="Where is the answer?",
        data_source="nq",
        ability="fact-reasoning",
        golden_answers=["there"],
        reward_model=None,
        metadata=None,
    )
    record = downloader._process_search_row(row, "train")

    assert record["data_source"] == "nq"
    assert record["prompt"][1]["content"] == "Where is the answer?"
    assert record["extra_info"]["need_tools_kwargs"] is True
    assert record["env_kwargs"]["ground_truth"] == ["there"]


def test_zip_member_cannot_escape_destination():
    downloader = load_downloader()

    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        downloader._safe_zip_target(Path("/tmp/dataset"), "../../outside.txt")


def test_webshop_download_uses_current_gdown_signature(monkeypatch, tmp_path):
    downloader = load_downloader()
    calls = {}
    fake_gdown = types.ModuleType("gdown")

    def download(*, id, output, quiet, resume):
        calls.update(id=id, output=output, quiet=quiet, resume=resume)
        destination = Path(output)
        destination.write_bytes(b"test")
        return str(destination)

    fake_gdown.download = download
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    destination = tmp_path / "items.json"
    downloader._download_google_drive_file("file-id", destination, force=False)

    assert calls == {
        "id": "file-id",
        "output": str(destination),
        "quiet": False,
        "resume": True,
    }


def test_webshop_download_prefers_public_huggingface_mirror(monkeypatch, tmp_path):
    downloader = load_downloader()
    calls = []

    def download_hf_file(repo_id, filename, destination_dir, repo_type="dataset", force=False):
        calls.append((repo_id, filename, destination_dir, repo_type, force))
        destination = destination_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"test")
        return destination

    def download_google_drive_file(*args, **kwargs):
        raise AssertionError("Google Drive should not be used when the mirror works")

    monkeypatch.setattr(downloader, "_download_hf_file", download_hf_file)
    monkeypatch.setattr(downloader, "_download_google_drive_file", download_google_drive_file)

    downloader.download_webshop(tmp_path, tmp_path / "weights", size="small", force=False)

    assert [call[0:2] for call in calls] == [
        ("HongbangYuan/webshop", "items_shuffle_1000.json"),
        ("HongbangYuan/webshop", "items_ins_v2_1000.json"),
        ("HongbangYuan/webshop", "items_human_ins.json"),
    ]
