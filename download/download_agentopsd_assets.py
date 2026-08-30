#!/usr/bin/env python3
"""Download AgentOPSD models and task data into fixed local directories.

The script only performs downloads when executed. Importing it or running with
``--dry-run`` does not access the network.

Examples:
    python download/download_agentopsd_assets.py --dry-run
    python download/download_agentopsd_assets.py --task models alfworld
    python download/download_agentopsd_assets.py --task all --webshop-size all

The default output layout is:

    /home/shuixia/users/houguoqiang/code/datasets/
    /home/shuixia/users/houguoqiang/code/weights/

Use ``--help`` for task-specific and output-directory overrides.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/home/shuixia/users/houguoqiang/code/datasets")
DEFAULT_WEIGHTS_DIR = Path("/home/shuixia/users/houguoqiang/code/weights")

TASK_CHOICES = ("models", "alfworld", "geometry3k", "search", "webshop", "all")
ALL_TASKS = ("models", "alfworld", "geometry3k", "search", "webshop")

QWEN_MODELS = (
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
)
RETRIEVER_MODEL = ("intfloat/e5-base-v2", "e5-base-v2")

ALFWORLD_ARCHIVES = (
    (
        "json_2.1.1_json.zip",
        "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    ),
    (
        "json_2.1.1_pddl.zip",
        "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    ),
    (
        "json_2.1.2_tw-pddl.zip",
        "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip",
    ),
)
ALFWORLD_MRCNN_URL = (
    "https://github.com/alfworld/alfworld/releases/download/0.2.2/"
    "mrcnn_alfred_objects_sep13_004.pth"
)
ALFWORLD_MRCNN_NAME = "mrcnn_alfred_objects_sep13_004.pth"

SEARCH_QA_REPO = "PeterJinGo/nq_hotpotqa_train"
SEARCH_INDEX_REPO = "PeterJinGo/wiki-18-e5-index"
SEARCH_CORPUS_REPO = "PeterJinGo/wiki-18-corpus"

WEBSHOP_FILE_IDS = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O",
}
WEBSHOP_HF_REPO = "HongbangYuan/webshop"
WEBSHOP_SMALL_FILES = (
    "items_shuffle_1000.json",
    "items_ins_v2_1000.json",
    "items_human_ins.json",
)
WEBSHOP_ALL_FILES = tuple(WEBSHOP_FILE_IDS)


@dataclass(frozen=True)
class ManifestEntry:
    """One logical asset in the download plan."""

    task: str
    kind: str
    name: str
    source: str
    relative_path: str


def normalize_tasks(tasks: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize CLI task names and expand ``all``."""

    if isinstance(tasks, str):
        requested = [tasks]
    else:
        requested = list(tasks or ("all",))

    unknown = sorted(set(requested) - set(TASK_CHOICES))
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    if not requested or "all" in requested:
        return ALL_TASKS
    return tuple(task for task in ALL_TASKS if task in requested)


def build_manifest(tasks: Sequence[str] | str | None = None, webshop_size: str = "all") -> tuple[ManifestEntry, ...]:
    """Build a side-effect-free inventory of all assets to be downloaded."""

    if webshop_size not in {"small", "all"}:
        raise ValueError("webshop_size must be 'small' or 'all'")

    selected = normalize_tasks(tasks)
    entries: list[ManifestEntry] = []

    if "models" in selected:
        for repo_id, directory_name in QWEN_MODELS:
            entries.append(
                ManifestEntry(
                    task="models",
                    kind="weight",
                    name=repo_id,
                    source=f"hf://{repo_id}",
                    relative_path=directory_name,
                )
            )
        entries.append(
            ManifestEntry(
                task="models",
                kind="weight",
                name=RETRIEVER_MODEL[0],
                source=f"hf://{RETRIEVER_MODEL[0]}",
                relative_path=RETRIEVER_MODEL[1],
            )
        )

    if "alfworld" in selected:
        for filename, url in ALFWORLD_ARCHIVES:
            entries.append(
                ManifestEntry(
                    task="alfworld",
                    kind="dataset",
                    name=filename,
                    source=url,
                    relative_path=f"alfworld/{filename}",
                )
            )
        entries.append(
            ManifestEntry(
                task="alfworld",
                kind="weight",
                name=ALFWORLD_MRCNN_NAME,
                source=ALFWORLD_MRCNN_URL,
                relative_path=f"alfworld/{ALFWORLD_MRCNN_NAME}",
            )
        )

    if "geometry3k" in selected:
        entries.append(
            ManifestEntry(
                task="geometry3k",
                kind="dataset",
                name="hiyouga/geometry3k",
                source="hf://datasets/hiyouga/geometry3k",
                relative_path="geometry3k/",
            )
        )
        entries.append(
            ManifestEntry(
                task="geometry3k",
                kind="dataset",
                name="AgentOPSD text parquet",
                source="generated from hiyouga/geometry3k",
                relative_path="verl-agent/text/{train,test}.parquet",
            )
        )

    if "search" in selected:
        entries.extend(
            (
                ManifestEntry(
                    task="search",
                    kind="dataset",
                    name="Search QA source parquet",
                    source=f"hf://datasets/{SEARCH_QA_REPO}",
                    relative_path="searchR1/nq_hotpotqa_train/{train,test}.parquet",
                ),
                ManifestEntry(
                    task="search",
                    kind="dataset",
                    name="Search QA processed parquet",
                    source=f"generated from {SEARCH_QA_REPO}",
                    relative_path="searchR1_processed_direct/{train,test}.parquet",
                ),
                ManifestEntry(
                    task="search",
                    kind="dataset",
                    name="wiki-18 FAISS index",
                    source=f"hf://datasets/{SEARCH_INDEX_REPO}",
                    relative_path="searchR1/e5_Flat.index",
                ),
                ManifestEntry(
                    task="search",
                    kind="dataset",
                    name="wiki-18 corpus",
                    source=f"hf://datasets/{SEARCH_CORPUS_REPO}",
                    relative_path="searchR1/wiki-18.jsonl",
                ),
            )
        )

    if "webshop" in selected:
        filenames = WEBSHOP_SMALL_FILES if webshop_size == "small" else WEBSHOP_ALL_FILES
        for filename in filenames:
            entries.append(
                ManifestEntry(
                    task="webshop",
                    kind="dataset",
                    name=filename,
                    source=f"hf://datasets/{WEBSHOP_HF_REPO}/{filename}",
                    relative_path=f"webshop/data/{filename}",
                )
            )

    return tuple(entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download AgentOPSD datasets and model weights without running training."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=TASK_CHOICES,
        default=["all"],
        metavar="TASK",
        help="Tasks to download: models, alfworld, geometry3k, search, webshop, or all (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory for datasets (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=f"Root directory for model weights (default: {DEFAULT_WEIGHTS_DIR}).",
    )
    parser.add_argument(
        "--webshop-size",
        choices=("small", "all"),
        default="all",
        help="Download the 1K WebShop subset or the full catalog (default: all).",
    )
    parser.add_argument(
        "--train-data-size",
        type=int,
        default=16,
        help="Number of Geometry3K train rows used for AgentOPSD text parquet (default: 16).",
    )
    parser.add_argument(
        "--val-data-size",
        type=int,
        default=128,
        help="Number of Geometry3K test rows used for AgentOPSD text parquet (default: 128).",
    )
    parser.add_argument(
        "--keep-search-parts",
        action="store_true",
        help="Keep SearchR1 index parts after creating e5_Flat.index.",
    )
    parser.add_argument(
        "--keep-search-archive",
        action="store_true",
        help="Keep wiki-18.jsonl.gz after creating wiki-18.jsonl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload/recreate existing files. Use only when replacement is intended.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without creating directories or accessing the network.",
    )
    return parser


def _print_manifest(entries: Iterable[ManifestEntry], data_dir: Path, weights_dir: Path) -> None:
    for entry in entries:
        root = weights_dir if entry.kind == "weight" else data_dir
        print(f"[{entry.kind}] {entry.name}")
        print(f"  source: {entry.source}")
        print(f"  target: {root / entry.relative_path}")


def _copy_file_if_needed(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[skip] {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    print(f"[ready] {destination}")


def _download_http_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.is_file() and not force:
        print(f"[skip] {destination}")
        return destination

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("ALFWorld downloads require the requests package") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    os.replace(temporary, destination)
    print(f"[ready] {destination}")
    return destination


def _safe_zip_target(root: Path, member_name: str) -> Path:
    root_resolved = root.resolve()
    target = (root / member_name).resolve()
    try:
        common = os.path.commonpath((str(root_resolved), str(target)))
    except ValueError as exc:
        raise RuntimeError(f"Unsafe archive member: {member_name}") from exc
    if common != str(root_resolved):
        raise RuntimeError(f"Unsafe archive member: {member_name}")
    return target


def _extract_zip(archive: Path, destination: Path, force: bool = False) -> None:
    print(f"[extract] {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = _safe_zip_target(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _download_hf_snapshot(repo_id: str, destination: Path, repo_type: str, force: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Hugging Face downloads require huggingface_hub") from exc

    destination.mkdir(parents=True, exist_ok=True)
    print(f"[download] hf://{repo_type}/{repo_id} -> {destination}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(destination),
        force_download=force,
    )
    print(f"[ready] {destination}")


def _download_hf_file(
    repo_id: str,
    filename: str,
    destination_dir: Path,
    repo_type: str = "dataset",
    force: bool = False,
) -> Path:
    destination = destination_dir / filename
    if destination.is_file() and not force:
        print(f"[skip] {destination}")
        return destination

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Hugging Face downloads require huggingface_hub") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] hf://{repo_type}/{repo_id}/{filename}")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(destination_dir),
            force_download=force,
        )
    )
    if downloaded != destination and downloaded.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
    print(f"[ready] {destination}")
    return destination


def _merge_files(parts: Sequence[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    os.replace(temporary, destination)
    print(f"[ready] {destination}")


def _decompress_gzip(archive: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    print(f"[extract] {archive}")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    os.replace(temporary, destination)
    print(f"[ready] {destination}")


def _link_weight(source: Path, destination: Path, force: bool) -> None:
    if destination.is_symlink() or destination.exists():
        if not force:
            print(f"[skip] {destination}")
            return
        if destination.is_dir() and not destination.is_symlink():
            raise RuntimeError(f"Cannot replace directory with symlink: {destination}")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative_source)
    print(f"[link] {destination} -> {source}")


def _find_builtin_alfworld_file(filename: str) -> Path | None:
    candidates = [
        REPO_ROOT / "agent_system/environments/env_package/alfworld/alfworld/data" / filename,
    ]
    try:
        import alfworld.info as alfworld_info

        candidates.append(Path(alfworld_info.BUILTIN_DATA_PATH) / filename)
    except ImportError:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def download_models(data_dir: Path, weights_dir: Path, force: bool) -> None:
    del data_dir
    for repo_id, directory_name in QWEN_MODELS:
        _download_hf_snapshot(repo_id, weights_dir / directory_name, "model", force)
    _download_hf_snapshot(RETRIEVER_MODEL[0], weights_dir / RETRIEVER_MODEL[1], "model", force)


def download_alfworld(data_dir: Path, weights_dir: Path, force: bool) -> None:
    data_root = data_dir / "alfworld"
    archive_root = data_root / ".downloads"
    archive_root.mkdir(parents=True, exist_ok=True)

    for filename, url in ALFWORLD_ARCHIVES:
        archive = archive_root / filename
        _download_http_file(url, archive, force=force)
        _extract_zip(archive, data_root, force=force)
        archive.unlink()

    detector = weights_dir / "alfworld" / ALFWORLD_MRCNN_NAME
    _download_http_file(ALFWORLD_MRCNN_URL, detector, force=force)
    _link_weight(detector, data_root / "detectors" / "mrcnn.pth", force=force)

    for filename in ("alfred.pddl", "alfred.twl2"):
        source = _find_builtin_alfworld_file(filename)
        if source is None:
            raise RuntimeError(f"Cannot find built-in ALFWorld file: {filename}")
        _copy_file_if_needed(source, data_root / "logic" / filename, force=force)

    try:
        archive_root.rmdir()
    except OSError:
        pass


def _geometry_parquet_files(geometry_dir: Path) -> tuple[Path, Path]:
    data_dir = geometry_dir / "data"
    train_files = sorted(data_dir.glob("train-*.parquet")) + sorted(geometry_dir.glob("train-*.parquet"))
    test_files = sorted(data_dir.glob("test-*.parquet")) + sorted(geometry_dir.glob("test-*.parquet"))
    if not train_files or not test_files:
        raise RuntimeError(f"Geometry3K parquet files not found under {geometry_dir}")
    return train_files[0], test_files[0]


def _write_dataset_atomic(dataset: object, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    dataset.to_parquet(str(temporary))
    os.replace(temporary, destination)
    print(f"[ready] {destination}")


def prepare_geometry_text_data(
    geometry_dir: Path,
    output_dir: Path,
    train_data_size: int,
    val_data_size: int,
    force: bool,
) -> None:
    if train_data_size < 1 or val_data_size < 1:
        raise ValueError("Geometry3K data sizes must be positive")

    train_output = output_dir / "train.parquet"
    test_output = output_dir / "test.parquet"
    if train_output.is_file() and test_output.is_file() and not force:
        print(f"[skip] {output_dir}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Geometry3K preparation requires the datasets package") from exc

    train_file, test_file = _geometry_parquet_files(geometry_dir)
    dataset = load_dataset(
        "parquet",
        data_files={"train": str(train_file), "test": str(test_file)},
    )
    if train_data_size > len(dataset["train"]):
        raise ValueError(f"Requested {train_data_size} train rows, only {len(dataset['train'])} exist")
    if val_data_size > len(dataset["test"]):
        raise ValueError(f"Requested {val_data_size} test rows, only {len(dataset['test'])} exist")

    def process(example: dict, index: int) -> dict:
        del example
        return {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "train", "index": index},
        }

    def process_test(example: dict, index: int) -> dict:
        del example
        return {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "test", "index": index},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train = dataset["train"].select(range(train_data_size)).map(
        process,
        with_indices=True,
        remove_columns=dataset["train"].column_names,
    )
    test = dataset["test"].select(range(val_data_size)).map(
        process_test,
        with_indices=True,
        remove_columns=dataset["test"].column_names,
    )
    _write_dataset_atomic(train, train_output)
    _write_dataset_atomic(test, test_output)


def download_geometry3k(
    data_dir: Path,
    weights_dir: Path,
    train_data_size: int,
    val_data_size: int,
    force: bool,
) -> None:
    del weights_dir
    geometry_dir = data_dir / "geometry3k"
    _download_hf_snapshot("hiyouga/geometry3k", geometry_dir, "dataset", force)
    prepare_geometry_text_data(
        geometry_dir=geometry_dir,
        output_dir=data_dir / "verl-agent" / "text",
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        force=force,
    )


def _process_search_row(row: object, split: str) -> dict:
    question = row.get("question", "")
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    data_source = str(row.get("data_source", ""))
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": question},
        ],
        "ability": row.get("ability"),
        "reward_model": reward_model_data,
        "extra_info": {
            "index": row.name,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": tools_kwargs,
        },
        "metadata": row.get("metadata"),
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
        },
    }


def _prepare_search_qa(source_dir: Path, output_dir: Path, force: bool) -> None:
    if (
        (output_dir / "train.parquet").is_file()
        and (output_dir / "test.parquet").is_file()
        and not force
    ):
        print(f"[skip] {output_dir}")
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Search QA preparation requires pandas") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = source_dir / f"{split}.parquet"
        if not source.is_file():
            raise RuntimeError(f"Search QA source file not found: {source}")
        print(f"[process] {source}")
        raw = pd.read_parquet(source)
        records = [_process_search_row(row, split) for _, row in raw.iterrows()]
        processed = pd.DataFrame.from_records(records)
        _write_dataframe_atomic(processed, output_dir / f"{split}.parquet")


def _write_dataframe_atomic(dataframe: object, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    dataframe.to_parquet(temporary, index=False)
    os.replace(temporary, destination)
    print(f"[ready] {destination}")


def _download_search_qa(data_dir: Path, force: bool) -> None:
    source_dir = data_dir / "searchR1" / "nq_hotpotqa_train"
    for split in ("train", "test"):
        _download_hf_file(SEARCH_QA_REPO, f"{split}.parquet", source_dir, force=force)
    _prepare_search_qa(source_dir, data_dir / "searchR1_processed_direct", force=force)


def _download_search_retrieval(
    data_dir: Path,
    force: bool,
    keep_parts: bool,
    keep_archive: bool,
) -> None:
    retrieval_dir = data_dir / "searchR1"
    index = retrieval_dir / "e5_Flat.index"
    if not index.is_file() or force:
        part_paths = [
            _download_hf_file(SEARCH_INDEX_REPO, "part_aa", retrieval_dir, force=force),
            _download_hf_file(SEARCH_INDEX_REPO, "part_ab", retrieval_dir, force=force),
        ]
        _merge_files(part_paths, index)
        if not keep_parts:
            for part in part_paths:
                part.unlink(missing_ok=True)
    else:
        print(f"[skip] {index}")

    corpus = retrieval_dir / "wiki-18.jsonl"
    archive = retrieval_dir / "wiki-18.jsonl.gz"
    if not corpus.is_file() or force:
        _download_hf_file(SEARCH_CORPUS_REPO, "wiki-18.jsonl.gz", retrieval_dir, force=force)
        _decompress_gzip(archive, corpus)
        if not keep_archive:
            archive.unlink(missing_ok=True)
    else:
        print(f"[skip] {corpus}")


def download_search(
    data_dir: Path,
    weights_dir: Path,
    force: bool,
    keep_parts: bool,
    keep_archive: bool,
) -> None:
    del weights_dir
    _download_search_qa(data_dir, force=force)
    _download_search_retrieval(
        data_dir=data_dir,
        force=force,
        keep_parts=keep_parts,
        keep_archive=keep_archive,
    )


def _download_google_drive_file(file_id: str, destination: Path, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"[skip] {destination}")
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("WebShop downloads require gdown; install it in the active environment") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    print(f"[download] gdrive://{file_id}")
    downloaded = gdown.download(
        id=file_id,
        output=str(destination),
        quiet=False,
        resume=True,
    )
    if not downloaded or not destination.is_file():
        raise RuntimeError(f"Google Drive download failed: {file_id}")
    print(f"[ready] {destination}")


def _download_webshop_file(filename: str, destination: Path, force: bool) -> Path:
    """Download WebShop data from a public mirror, then try the original source."""

    try:
        return _download_hf_file(
            WEBSHOP_HF_REPO,
            filename,
            destination.parent,
            force=force,
        )
    except Exception as mirror_error:
        print(f"[fallback] WebShop mirror unavailable for {filename}: {mirror_error}")
        try:
            _download_google_drive_file(
                WEBSHOP_FILE_IDS[filename],
                destination,
                force=force,
            )
        except Exception as drive_error:
            raise RuntimeError(
                f"Unable to download WebShop file {filename} from the public mirror "
                "or Google Drive"
            ) from drive_error
        return destination


def download_webshop(data_dir: Path, weights_dir: Path, size: str, force: bool) -> None:
    del weights_dir
    filenames = WEBSHOP_SMALL_FILES if size == "small" else WEBSHOP_ALL_FILES
    target_dir = data_dir / "webshop" / "data"
    for filename in filenames:
        _download_webshop_file(filename, target_dir / filename, force=force)


def download_selected(
    tasks: Sequence[str],
    data_dir: Path,
    weights_dir: Path,
    webshop_size: str,
    train_data_size: int,
    val_data_size: int,
    force: bool,
    keep_search_parts: bool,
    keep_search_archive: bool,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if "models" in tasks:
        download_models(data_dir, weights_dir, force=force)
    if "alfworld" in tasks:
        download_alfworld(data_dir, weights_dir, force=force)
    if "geometry3k" in tasks:
        download_geometry3k(
            data_dir,
            weights_dir,
            train_data_size=train_data_size,
            val_data_size=val_data_size,
            force=force,
        )
    if "search" in tasks:
        download_search(
            data_dir,
            weights_dir,
            force=force,
            keep_parts=keep_search_parts,
            keep_archive=keep_search_archive,
        )
    if "webshop" in tasks:
        download_webshop(data_dir, weights_dir, size=webshop_size, force=force)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = normalize_tasks(args.task)
    entries = build_manifest(tasks, webshop_size=args.webshop_size)

    if args.dry_run:
        print(f"data_dir={args.data_dir}")
        print(f"weights_dir={args.weights_dir}")
        _print_manifest(entries, args.data_dir, args.weights_dir)
        return 0

    download_selected(
        tasks=tasks,
        data_dir=args.data_dir,
        weights_dir=args.weights_dir,
        webshop_size=args.webshop_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        force=args.force,
        keep_search_parts=args.keep_search_parts,
        keep_search_archive=args.keep_search_archive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
