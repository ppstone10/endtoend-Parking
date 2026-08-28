"""专家数据能力校准、硬预算与可中断恢复。"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any, Callable, Iterable

from sim import Maneuver, NoiseLevel, TaskSampler, TaskType, VehicleConfig

from .build import expert_maneuvers
from .components import build_task_components
from .generator import DatasetGenerator, TaskGenerationError


_TERMINAL_STATUSES = {"success", "failed", "task_budget_exceeded"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


@dataclass(frozen=True)
class CalibrationSettings:
    """一次能力校准的稳定参数。"""

    samples_per_cell: int = 3
    seed: int = 20260824
    max_retries: int = 2
    task_budget_s: float = 30.0
    probe_full_capability: bool = False

    def __post_init__(self) -> None:
        if self.samples_per_cell <= 0:
            raise ValueError("samples_per_cell 必须为正")
        if self.seed < 0:
            raise ValueError("seed 不能为负")
        if self.max_retries < 0:
            raise ValueError("max_retries 不能为负")
        if not math.isfinite(self.task_budget_s) or self.task_budget_s <= 0.0:
            raise ValueError("task_budget_s 必须为有限正数")
        if not isinstance(self.probe_full_capability, bool):
            raise ValueError("probe_full_capability 必须为布尔值")


@dataclass(frozen=True)
class CalibrationCase:
    """一个场景×任务类型下可独立恢复的校准 case。"""

    case_id: str
    scene_name: str
    task_type: str
    ordinal: int
    maneuver: str
    noise_level: str
    adjacent_occupancy: int

    @property
    def cell_id(self) -> str:
        return f"{self.scene_name}/{self.task_type}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationResult:
    """单个校准 case 的终态证据。"""

    case_id: str
    scene_name: str
    task_type: str
    status: str
    duration_s: float
    attempts: int
    failure_count: int
    failure_reasons: dict[str, int]
    task_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _TERMINAL_STATUSES:
            raise ValueError(f"非法校准终态 {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationResult":
        return cls(**payload)


def build_calibration_cases(
    sampler: TaskSampler,
    samples_per_cell: int,
    *,
    probe_full_capability: bool = False,
) -> tuple[CalibrationCase, ...]:
    """构造与数据配额无关的已准入或原始几何能力校准计划。"""
    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell 必须为正")
    cases: list[CalibrationCase] = []
    noises = tuple(NoiseLevel)
    for cell in sampler.capability_matrix():
        maneuvers = (
            (Maneuver.FORWARD, Maneuver.REVERSE)
            if probe_full_capability
            else expert_maneuvers(cell.scene_name, cell.task_type)
        )
        if not cell.supported or not maneuvers:
            continue
        occupancies = sampler.adjacent_occupancy_levels(
            cell.scene_name, cell.task_type
        )
        for ordinal in range(samples_per_cell):
            cases.append(
                CalibrationCase(
                    case_id=f"{cell.scene_name}__{cell.task_type.value}__{ordinal:03d}",
                    scene_name=cell.scene_name,
                    task_type=cell.task_type.value,
                    ordinal=ordinal,
                    maneuver=maneuvers[ordinal % len(maneuvers)].value,
                    noise_level=noises[ordinal % len(noises)].value,
                    adjacent_occupancy=occupancies[ordinal % len(occupancies)],
                )
            )
    return tuple(cases)


def _failure_code(exc: BaseException) -> str:
    return exc.code if isinstance(exc, TaskGenerationError) else type(exc).__name__


def _execute_case_worker(
    case_payload: dict[str, Any],
    settings_payload: dict[str, Any],
    vehicle_payload: dict[str, Any],
    result_queue,
) -> None:
    """在独立进程中执行一个 case，并只返回小型结果证据。"""
    started = time.perf_counter()
    case = CalibrationCase(**case_payload)
    settings = CalibrationSettings(**settings_payload)
    vehicle_config = VehicleConfig(**vehicle_payload)
    sampler = TaskSampler(
        seed=settings.seed,
        vehicle_length=vehicle_config.length,
        vehicle_width=vehicle_config.width,
        collision_margin=vehicle_config.collision_margin,
    )
    generator = DatasetGenerator(
        component_factory=lambda task: build_task_components(task, vehicle_config)
    )
    reasons: Counter[str] = Counter()
    last_error: str | None = None
    task_id: str | None = None
    attempts = 0
    try:
        for attempt in range(settings.max_retries + 1):
            attempts = attempt + 1
            sample_index = case.ordinal + attempt * settings.samples_per_cell
            try:
                task = sampler.sample(
                    case.scene_name,
                    TaskType(case.task_type),
                    sample_index=sample_index,
                    maneuver=Maneuver(case.maneuver),
                    adjacent_occupancy=case.adjacent_occupancy,
                    noise_level=NoiseLevel(case.noise_level),
                )
                task_id = task.task_id
                generator.generate([task])
                result_queue.put(
                    CalibrationResult(
                        case_id=case.case_id,
                        scene_name=case.scene_name,
                        task_type=case.task_type,
                        status="success",
                        duration_s=time.perf_counter() - started,
                        attempts=attempts,
                        failure_count=sum(reasons.values()),
                        failure_reasons=dict(sorted(reasons.items())),
                        task_id=task_id,
                    ).to_dict()
                )
                return
            except Exception as exc:
                reasons[_failure_code(exc)] += 1
                last_error = str(exc)
        result_queue.put(
            CalibrationResult(
                case_id=case.case_id,
                scene_name=case.scene_name,
                task_type=case.task_type,
                status="failed",
                duration_s=time.perf_counter() - started,
                attempts=attempts,
                failure_count=sum(reasons.values()),
                failure_reasons=dict(sorted(reasons.items())),
                task_id=task_id,
                error=last_error,
            ).to_dict()
        )
    except BaseException as exc:
        reasons[type(exc).__name__] += 1
        result_queue.put(
            CalibrationResult(
                case_id=case.case_id,
                scene_name=case.scene_name,
                task_type=case.task_type,
                status="failed",
                duration_s=time.perf_counter() - started,
                attempts=attempts,
                failure_count=sum(reasons.values()),
                failure_reasons=dict(sorted(reasons.items())),
                task_id=task_id,
                error=f"worker 异常：{exc}",
            ).to_dict()
        )


def run_case_with_budget(
    case: CalibrationCase,
    settings: CalibrationSettings,
    vehicle_config: VehicleConfig,
) -> CalibrationResult:
    """在硬墙钟预算内运行一个 case，超时时回收独立进程。"""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_execute_case_worker,
        args=(
            case.to_dict(),
            asdict(settings),
            vehicle_config.to_metadata(),
            result_queue,
        ),
    )
    started = time.perf_counter()
    process.start()
    try:
        process.join(settings.task_budget_s)
        duration = time.perf_counter() - started
        if process.is_alive():
            process.terminate()
            process.join(2.0)
            if process.is_alive():
                process.kill()
                process.join()
            return CalibrationResult(
                case_id=case.case_id,
                scene_name=case.scene_name,
                task_type=case.task_type,
                status="task_budget_exceeded",
                duration_s=duration,
                attempts=0,
                failure_count=1,
                failure_reasons={"task_budget_exceeded": 1},
                error=f"单 case 超过 {settings.task_budget_s:.1f}s 总预算",
            )
        try:
            payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            return CalibrationResult(
                case_id=case.case_id,
                scene_name=case.scene_name,
                task_type=case.task_type,
                status="failed",
                duration_s=duration,
                attempts=0,
                failure_count=1,
                failure_reasons={"worker_no_result": 1},
                error=f"worker 退出码 {process.exitcode}，但未返回结果",
            )
        return CalibrationResult.from_dict(payload)
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
            process.join(2.0)
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    fields = [
        "scene_name",
        "task_type",
        "planned",
        "completed",
        "success",
        "failed",
        "task_budget_exceeded",
        "failure_count",
        "completion_rate",
        "success_rate",
        "total_duration_s",
        "mean_duration_s",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _aggregate_report(
    cases: tuple[CalibrationCase, ...],
    records: dict[str, CalibrationResult],
    *,
    status: str,
    output: Path,
) -> dict[str, Any]:
    cell_cases: dict[tuple[str, str], list[CalibrationCase]] = {}
    for case in cases:
        cell_cases.setdefault((case.scene_name, case.task_type), []).append(case)
    cell_rows: list[dict[str, Any]] = []
    for (scene_name, task_type), planned_cases in cell_cases.items():
        results = [records[case.case_id] for case in planned_cases if case.case_id in records]
        counts = Counter(result.status for result in results)
        duration = sum(result.duration_s for result in results)
        cell_rows.append(
            {
                "scene_name": scene_name,
                "task_type": task_type,
                "planned": len(planned_cases),
                "completed": len(results),
                "success": counts["success"],
                "failed": counts["failed"],
                "task_budget_exceeded": counts["task_budget_exceeded"],
                "failure_count": sum(result.failure_count for result in results),
                "completion_rate": len(results) / len(planned_cases),
                "success_rate": counts["success"] / len(planned_cases),
                "total_duration_s": duration,
                "mean_duration_s": duration / len(results) if results else 0.0,
            }
        )
    completed = len(records)
    status_counts = Counter(record.status for record in records.values())
    reasons: Counter[str] = Counter()
    for record in records.values():
        reasons.update(record.failure_reasons)
    return {
        "schema_version": 1,
        "status": status,
        "is_partial": completed < len(cases),
        "output": str(output),
        "planned_cases": len(cases),
        "completed_cases": completed,
        "remaining_cases": len(cases) - completed,
        "status_counts": dict(sorted(status_counts.items())),
        "failure_reasons": dict(sorted(reasons.items())),
        "cells": cell_rows,
        "cases": [
            records[case.case_id].to_dict()
            for case in cases
            if case.case_id in records
        ],
        "updated_at": _now_iso(),
    }


def _load_records(
    checkpoint_root: Path, cases: tuple[CalibrationCase, ...]
) -> dict[str, CalibrationResult]:
    records: dict[str, CalibrationResult] = {}
    expected = {case.case_id: case for case in cases}
    if not checkpoint_root.exists():
        return records
    for path in sorted(checkpoint_root.glob("*.json")):
        record = CalibrationResult.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        case = expected.get(record.case_id)
        if case is None:
            raise RuntimeError(f"检查点包含计划外 case：{record.case_id}")
        if (record.scene_name, record.task_type) != (case.scene_name, case.task_type):
            raise RuntimeError(f"检查点单元与计划不匹配：{path}")
        records[record.case_id] = record
    return records


CaseRunner = Callable[
    [CalibrationCase, CalibrationSettings, VehicleConfig], CalibrationResult
]


def run_calibration(
    output: str | Path,
    settings: CalibrationSettings,
    vehicle_config: VehicleConfig,
    *,
    cases: Iterable[CalibrationCase] | None = None,
    case_runner: CaseRunner = run_case_with_budget,
) -> dict[str, Any]:
    """运行或恢复校准；case 终态落盘后才计入完成数。"""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_cases = tuple(cases) if cases is not None else build_calibration_cases(
        TaskSampler(
            seed=settings.seed,
            vehicle_length=vehicle_config.length,
            vehicle_width=vehicle_config.width,
            collision_margin=vehicle_config.collision_margin,
        ),
        settings.samples_per_cell,
        probe_full_capability=settings.probe_full_capability,
    )
    if not resolved_cases:
        raise ValueError("校准计划不能为空")
    identity = {
        "schema_version": 1,
        "settings": asdict(settings),
        "vehicle_model": vehicle_config.to_metadata(),
        "cases": [case.to_dict() for case in resolved_cases],
    }
    identity_path = output_path / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("现有校准目录身份与本次参数、车辆模型或 case 计划不一致")
    else:
        _write_json_atomic(identity_path, identity)

    checkpoint_root = output_path / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    records = _load_records(checkpoint_root, resolved_cases)
    state_path = output_path / "run_state.json"
    report_path = output_path / "report.json"
    csv_path = output_path / "cells.csv"
    previous_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    started_at = previous_state.get("started_at", _now_iso())
    run_started_at = _now_iso()

    def persist(status: str, current_case: str | None) -> dict[str, Any]:
        report = _aggregate_report(
            resolved_cases, records, status=status, output=output_path
        )
        _write_json_atomic(report_path, report)
        _write_csv_atomic(csv_path, report["cells"])
        timestamp = _now_iso()
        _write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "status": status,
                "started_at": started_at,
                "last_started_at": run_started_at,
                "updated_at": timestamp,
                "finished_at": (
                    timestamp if status in {"completed", "interrupted", "failed"} else None
                ),
                "completed_cases": len(records),
                "planned_cases": len(resolved_cases),
                "current_case": current_case,
                "is_partial": len(records) < len(resolved_cases),
            },
        )
        return report

    persist("running", None)
    try:
        for case in resolved_cases:
            if case.case_id in records:
                continue
            persist("running", case.case_id)
            try:
                result = case_runner(case, settings, vehicle_config)
            except KeyboardInterrupt:
                persist("interrupted", case.case_id)
                raise
            except Exception as exc:
                result = CalibrationResult(
                    case_id=case.case_id,
                    scene_name=case.scene_name,
                    task_type=case.task_type,
                    status="failed",
                    duration_s=0.0,
                    attempts=0,
                    failure_count=1,
                    failure_reasons={"case_runner_error": 1},
                    error=str(exc),
                )
            if (
                result.case_id != case.case_id
                or result.scene_name != case.scene_name
                or result.task_type != case.task_type
            ):
                raise RuntimeError(f"case runner 返回了不匹配结果：{case.case_id}")
            _write_json_atomic(
                checkpoint_root / f"{case.case_id}.json", result.to_dict()
            )
            records[case.case_id] = result
            persist("running", None)
    except KeyboardInterrupt:
        raise
    except BaseException:
        persist("failed", None)
        raise
    return persist("completed", None)
