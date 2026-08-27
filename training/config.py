"""严格、安全的 YAML 训练运行配置。"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

from .trainer import TrainerConfig


@dataclass(frozen=True)
class TrainingRunConfig:
    """一次可复现训练运行所需的全部输入。"""

    source: Path
    model_name: str
    model_config: dict[str, Any]
    train_data: Path
    val_data: Path
    batch_size: int
    trainer: TrainerConfig
    output_dir: Path
    resume_from: Path | None = None


def load_training_run_config(path: str | Path) -> TrainingRunConfig:
    """使用 PyYAML SafeLoader 读取并严格校验训练配置。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 环境错误由 CLI 呈现
        raise RuntimeError("读取训练 YAML 需要安装 PyYAML") from exc

    source = Path(path).resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"训练 YAML 无法读取：{exc}") from exc

    root = _mapping(raw, "根配置")
    _reject_unknown(root, {"model", "data", "training", "output", "resume_from"}, "根配置")
    model = _required_mapping(root, "model")
    data = _required_mapping(root, "data")
    training = _required_mapping(root, "training")
    output = _required_mapping(root, "output")

    _reject_unknown(model, {"name", "config"}, "model")
    _reject_unknown(data, {"train", "val", "batch_size"}, "data")
    trainer_fields = {item.name for item in fields(TrainerConfig)} - {"checkpoint_dir"}
    _reject_unknown(training, trainer_fields, "training")
    _reject_unknown(output, {"directory"}, "output")

    model_name = _required_string(model, "name", "model")
    model_config = dict(_mapping(model.get("config", {}), "model.config"))
    try:
        json.dumps(model_config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("model.config 必须只包含有限、可序列化值") from exc
    train_data = _resolve_existing_file(source.parent, data, "train")
    val_data = _resolve_existing_file(source.parent, data, "val")
    batch_size = data.get("batch_size", 8)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("data.batch_size 必须为正整数")

    output_dir = _resolve_path(
        source.parent, _required_string(output, "directory", "output")
    )
    trainer_values = dict(training)
    trainer_values["checkpoint_dir"] = str(output_dir)
    try:
        trainer = TrainerConfig(**trainer_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training 配置无效：{exc}") from exc

    resume_value = root.get("resume_from")
    if resume_value is not None and not isinstance(resume_value, str):
        raise ValueError("resume_from 必须为路径字符串或 null")
    resume_from = None
    if resume_value:
        resume_from = _resolve_path(source.parent, resume_value)
        if not resume_from.is_file():
            raise ValueError(f"resume_from 不存在：{resume_from}")

    return TrainingRunConfig(
        source=source,
        model_name=model_name,
        model_config=model_config,
        train_data=train_data,
        val_data=val_data,
        batch_size=batch_size,
        trainer=trainer,
        output_dir=output_dir,
        resume_from=resume_from,
    )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location} 必须为字符串键映射")
    return value


def _required_mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in root:
        raise ValueError(f"根配置缺少 {key}")
    return _mapping(root[key], key)


def _required_string(root: Mapping[str, Any], key: str, location: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} 必须为非空字符串")
    return value


def _reject_unknown(
    root: Mapping[str, Any], allowed: set[str], location: str
) -> None:
    unknown = sorted(set(root) - allowed)
    if unknown:
        raise ValueError(f"{location} 包含未知字段：{', '.join(unknown)}")


def _resolve_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _resolve_existing_file(
    base: Path, root: Mapping[str, Any], key: str
) -> Path:
    path = _resolve_path(base, _required_string(root, key, "data"))
    if not path.is_file():
        raise ValueError(f"data.{key} 不存在：{path}")
    return path
