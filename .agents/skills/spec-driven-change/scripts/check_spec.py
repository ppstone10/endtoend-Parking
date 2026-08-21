#!/usr/bin/env python3
"""只检查显式传入的 Spec 文件，避免扫描全部历史规格。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_HEADINGS = (
    "## 元数据",
    "## 目标",
    "## 非目标",
    "## 边界与约束",
    "## 行为与验收",
    "## 追溯",
)
SPEC_ID = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-[0-9]{3}\b")
BEHAVIOR_ID = re.compile(
    r"^###\s+`?([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-[0-9]{3})`?",
    re.MULTILINE,
)
TRACE_ROW = re.compile(
    r"^\|\s*`?([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-[0-9]{3})`?\s*\|",
    re.MULTILINE,
)


def validate(path: Path) -> list[str]:
    if not path.is_file():
        return ["文件不存在"]
    text = path.read_text(encoding="utf-8")
    problems = [
        f"缺少章节：{heading}" for heading in REQUIRED_HEADINGS if heading not in text
    ]
    ids = set(SPEC_ID.findall(text))
    if not ids:
        problems.append("没有稳定规范 ID（例如 AUTH-SESSION-001）")
    behavior_ids = set(BEHAVIOR_ID.findall(text))
    trace_ids = set(TRACE_ROW.findall(text))
    for spec_id in sorted(behavior_ids - trace_ids):
        problems.append(f"行为 {spec_id} 尚未进入追溯表")
    if "<功能名称>" in text or "<DOMAIN-TOPIC" in text:
        problems.append("仍包含模板占位符")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="检查指定 Spec 的最小结构与追溯映射")
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.files:
        problems = validate(path)
        if problems:
            failed = True
            print(f"FAIL: {path}")
            for problem in problems:
                print(f"- {problem}")
        else:
            print(f"PASS: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
