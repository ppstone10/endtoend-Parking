"""从旧 schema v2 专家检查点建立少量、可追溯的可视化参考归档。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DatasetGenerator, render_sample_overlay, summarize_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "top_level": [
            {
                "name": child.name,
                "file_count": (
                    1
                    if child.is_file()
                    else sum(path.is_file() for path in child.rglob("*"))
                ),
                "total_bytes": (
                    child.stat().st_size
                    if child.is_file()
                    else sum(
                        path.stat().st_size
                        for path in child.rglob("*")
                        if path.is_file()
                    )
                ),
            }
            for child in sorted(root.iterdir(), key=lambda path: path.name)
        ],
    }


def _require_within(path: Path, root: Path, field_name: str) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field_name} 必须位于 {root} 内：{resolved}")
    return resolved


def create_visual_reference_archive(
    selection_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """按显式选择清单建立 NPZ、统计、PNG、哈希和来源 manifest。"""
    selection_file = Path(selection_path).resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    cleanup_root = Path(selection["cleanup_root"]).resolve()
    if not cleanup_root.is_dir():
        raise ValueError(f"cleanup_root 不存在：{cleanup_root}")
    output = _require_within(Path(output_dir), cleanup_root, "output_dir")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"归档目录必须不存在或为空：{output}")
    entries = selection.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection.samples 必须是非空列表")

    before_cleanup = _directory_inventory(cleanup_root)
    source_cache: dict[Path, dict[str, Any]] = {}
    source_hashes: dict[Path, str] = {}
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        source = _require_within(
            cleanup_root / str(entry["source"]), cleanup_root, "sample.source"
        )
        if source == output or output in source.parents:
            raise ValueError("归档来源不能位于输出目录内")
        if source not in source_cache:
            source_cache[source] = DatasetGenerator.load(source)
        archive = source_cache[source]
        if int(archive.get("schema_version", 0)) != 2:
            raise ValueError(f"可视化参考只接受 schema v2：{source}")
        index = int(entry["index"])
        if not 0 <= index < int(archive["bevs"].shape[0]):
            raise IndexError(f"样本索引越界：{source}#{index}")
        source_hashes.setdefault(source, _sha256(source))
        chosen.append((entry, archive))

    first_archive = chosen[0][1]
    bev_meta = first_archive["bev_meta"]
    dt = float(np.asarray(first_archive["dt"]).reshape(-1)[0])
    for _, archive in chosen[1:]:
        if archive["bev_meta"] != bev_meta:
            raise ValueError("所选样本的 BEV 元数据不一致")
        current_dt = float(np.asarray(archive["dt"]).reshape(-1)[0])
        if not np.isclose(current_dt, dt):
            raise ValueError("所选样本的轨迹 dt 不一致")

    output.mkdir(parents=True, exist_ok=True)
    inspection_dir = output / "inspection"
    inspection_dir.mkdir()
    sample_count = len(chosen)
    max_horizon = max(
        int(np.count_nonzero(archive["masks"][int(entry["index"])]))
        for entry, archive in chosen
    )
    bevs = np.stack(
        [archive["bevs"][int(entry["index"])] for entry, archive in chosen]
    )
    goals = np.stack(
        [archive["goals"][int(entry["index"])] for entry, archive in chosen]
    )
    states = np.stack(
        [archive["states"][int(entry["index"])] for entry, archive in chosen]
    )
    trajs = np.zeros((sample_count, max_horizon, 3), dtype=np.float32)
    masks = np.zeros((sample_count, max_horizon), dtype=np.float32)
    archived_metadata: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    for archive_index, (entry, archive) in enumerate(chosen):
        source = _require_within(
            cleanup_root / str(entry["source"]), cleanup_root, "sample.source"
        )
        source_index = int(entry["index"])
        horizon = int(np.count_nonzero(archive["masks"][source_index]))
        trajs[archive_index, :horizon] = archive["trajs"][source_index, :horizon]
        masks[archive_index, :horizon] = 1.0
        metadata = copy.deepcopy(archive["task_meta"][source_index])
        metadata.setdefault("dataset", {})["visual_archive_source"] = {
            "source": source.relative_to(cleanup_root).as_posix(),
            "source_index": source_index,
            "source_sha256": source_hashes[source],
            "selection_reasons": list(entry.get("reasons", [])),
        }
        archived_metadata.append(metadata)
        sample_manifest.append(
            {
                "archive_index": archive_index,
                "source": source.relative_to(cleanup_root).as_posix(),
                "source_index": source_index,
                "source_sha256": source_hashes[source],
                "task_id": metadata.get("task_id"),
                "scene_name": metadata.get("scene_name"),
                "task_type": metadata.get("task_type"),
                "maneuver": metadata.get("difficulty", {}).get("maneuver"),
                "vehicle_model_version": metadata.get("dataset", {})
                .get("vehicle_model", {})
                .get("model_version"),
                "selection_reasons": list(entry.get("reasons", [])),
            }
        )

    archive_path = output / "reference_samples.npz"
    np.savez_compressed(
        archive_path,
        schema_version=np.asarray(2, dtype=np.uint16),
        bev_meta=np.asarray(
            DatasetGenerator._encode_metadata(bev_meta, "bev_meta"), dtype=np.str_
        ),
        task_meta=np.asarray(
            [
                DatasetGenerator._encode_metadata(metadata, "task_meta")
                for metadata in archived_metadata
            ],
            dtype=np.str_,
        ),
        bevs=bevs,
        goals=goals,
        states=states,
        trajs=trajs,
        masks=masks,
        dt=np.asarray([dt]),
    )

    loaded = DatasetGenerator.load(archive_path)
    if int(loaded["bevs"].shape[0]) != sample_count:
        raise RuntimeError("归档重读后的样本数不一致")
    summary = summarize_dataset(loaded)
    summary_path = inspection_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    image_paths: list[Path] = []
    for index in range(sample_count):
        image_path = inspection_dir / f"sample_{index:02d}.png"
        render_sample_overlay(loaded, index, image_path)
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise RuntimeError(f"归档验收图未成功生成：{image_path}")
        image_paths.append(image_path)

    readme_path = output / "README.md"
    readme_path.write_text(
        "# 旧专家数据可视化参考归档\n\n"
        "本目录只保留少量真实 tracked_pivot_v3 样本，用于复核前进、倒车、"
        "换向和原地旋转可视化。它不是训练集，也不能恢复已删除的旧全量数据。\n\n"
        "- `reference_samples.npz`：安全可读的 schema v2 参考样本；\n"
        "- `inspection/`：当前代码生成的统计与逐样本 PNG；\n"
        "- `archive_manifest.json`：来源、哈希、选择理由和清理前基线；\n"
        "- `checksums.sha256`：归档内容校验值。\n\n"
        "后续正式训练必须重新生成 tracked_pivot_v4 数据，不能将本目录作为训练输入。\n",
        encoding="utf-8",
    )

    content_paths = [archive_path, summary_path, readme_path, *image_paths]
    content_hashes = {
        path.relative_to(output).as_posix(): _sha256(path) for path in content_paths
    }
    manifest = {
        "archive_schema_version": 1,
        "created_date": str(selection.get("created_date", "2026-08-27")),
        "purpose": "historical expert trajectory visualization reference only",
        "cleanup_root": str(cleanup_root),
        "before_cleanup": before_cleanup,
        "sample_count": sample_count,
        "samples": sample_manifest,
        "content_sha256": content_hashes,
        "full_dataset_recoverable": False,
        "required_next_model_version": "tracked_pivot_v4",
    }
    manifest_path = output / "archive_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checksum_lines = [
        f"{digest}  {relative_path}"
        for relative_path, digest in sorted(content_hashes.items())
    ]
    checksum_lines.append(f"{_sha256(manifest_path)}  archive_manifest.json")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="建立旧专家数据可视化参考归档")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_visual_reference_archive(args.selection, args.output)
    print(
        f"可视化参考归档完成：{args.output}，"
        f"{manifest['sample_count']} 条真实样本"
    )


if __name__ == "__main__":
    main()
