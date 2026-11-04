"""旧源任务上的轮式教师校准：prepare/precheck/smoke/run/inspect。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification.wheeled_calibration import inspect_case, precheck, prepare_config, run_case, sha256_file


def write_new(path, data):
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def load_prepared(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    precheck(config)
    return config


def case_filename(ordinal, case):
    return f"{ordinal:03d}-{case['split']}-{case['index']}.json"


def run_formal(config, prepared, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root/"config.json", config)
    artifacts, rows = {}, []
    for ordinal, case in enumerate(config["cases"]):
        print(f"START {ordinal+1}/{len(config['cases'])} {case['split']}:{case['index']}", flush=True)
        result = run_case(config, case)
        path = root/case_filename(ordinal, case)
        write_new(path, result)
        artifacts[path.name] = sha256_file(path)
        rows.append(dict(case=case, planned=result["planned"], passed=result["passed"],
                         elapsed_seconds=result["elapsed_seconds"]))
        print(f"DONE planned={result['planned']} passed={result['passed']}", flush=True)
    report = dict(schema=config["schema"], status="completed", formal=True,
                  prepared_sha256=sha256_file(prepared), expected_cases=len(config["cases"]), rows=rows,
                  gate_passed=all(row["passed"] for row in rows), training_exported=False,
                  production_qualified=False)
    write_new(root/"report.json", report)
    artifacts["config.json"] = sha256_file(root/"config.json")
    artifacts["report.json"] = sha256_file(root/"report.json")
    write_new(root/"artifacts.json", artifacts)
    return dict(status="completed", cases=len(rows), gate_passed=report["gate_passed"])


def inspect_formal(config, prepared, output):
    root = Path(output)
    manifest = json.loads((root/"artifacts.json").read_text(encoding="utf-8"))
    expected = {"config.json", "report.json"} | {case_filename(i, c) for i, c in enumerate(config["cases"])}
    if set(manifest) != expected:
        raise ValueError("校准产物分母/路径集合不一致")
    for name, digest in manifest.items():
        if sha256_file(root/name) != digest:
            raise ValueError(f"校准产物SHA不一致：{name}")
    if json.loads((root/"config.json").read_text(encoding="utf-8")) != config:
        raise ValueError("运行配置与prepared不一致")
    report = json.loads((root/"report.json").read_text(encoding="utf-8"))
    if (report.get("schema") != config["schema"] or report.get("formal") is not True
            or report.get("status") != "completed" or report.get("prepared_sha256") != sha256_file(prepared)
            or report.get("expected_cases") != len(config["cases"]) or len(report.get("rows", [])) != len(config["cases"])):
        raise ValueError("校准报告未完整完成或身份不一致")
    passed = []
    for ordinal, case in enumerate(config["cases"]):
        record = json.loads((root/case_filename(ordinal, case)).read_text(encoding="utf-8"))
        quality = inspect_case(config, case, record)
        if report["rows"][ordinal]["case"] != case or report["rows"][ordinal]["passed"] != quality:
            raise ValueError("校准逐例汇总不一致")
        passed.append(quality)
    if report["gate_passed"] != all(passed):
        raise ValueError("校准联合门槛不一致")
    return dict(verified=True, cases=len(passed), passed=sum(passed), gate_passed=all(passed),
                training_exported=False, production_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "precheck", "smoke", "run", "inspect"))
    parser.add_argument("--prepared")
    parser.add_argument("--output")
    parser.add_argument("--source", action="append")
    parser.add_argument("--vehicle", default="configs/vehicles/wheeled_research_v1.json")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.output:
            parser.error("prepare要求--output新配置文件")
        sources = args.source or [f"data/task_dataset/tracked_pivot_v8_3000/{split}.npz" for split in ("train", "val")]
        config = prepare_config(sources, args.vehicle)
        write_new(args.output, config)
        print(json.dumps(dict(prepared=str(Path(args.output).resolve()), sha256=sha256_file(args.output),
                              cases=len(config["cases"]), frozen_files=len(config["frozen_files"])), ensure_ascii=False))
        return
    if not args.prepared:
        parser.error("本阶段要求--prepared")
    config = load_prepared(args.prepared)
    if args.phase == "precheck":
        result = dict(verified=True, cases=len(config["cases"]), formal_episodes=0)
    elif args.phase == "smoke":
        if not args.output:
            parser.error("smoke要求--output新证据文件")
        record = run_case(config, config["cases"][0], max_seconds=20.)
        audited = inspect_case(config, config["cases"][0], record)
        write_new(args.output, record)
        result = dict(kind="wheeled_calibration_smoke", formal=False, case=record["case"],
                      planned=record["planned"], quality_passed=audited,
                      elapsed_seconds=record["elapsed_seconds"])
    else:
        if not args.output:
            parser.error("run/inspect要求--output正式结果目录")
        result = (run_formal(config, args.prepared, args.output) if args.phase == "run"
                  else inspect_formal(config, args.prepared, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
