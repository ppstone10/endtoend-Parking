"""生成同数据不同 seed 的对照训练配置，用于量化训练随机性。

用法：
    & 'D:\\conda\\envs\\endtoend-parking\\python.exe' scripts/make_seed_variants.py \
        --template configs/training/net-v1-4scenes-v8data.yaml \
        --seeds 11,23 --output-dir configs/training/seed-variants
"""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE_SEED = "seed: 20260827"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--seeds", required=True, help="逗号分隔的随机种子")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="seed")
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    if TEMPLATE_SEED not in template:
        raise ValueError(f"模板中找不到 {TEMPLATE_SEED}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.seeds.split(","):
        seed = raw.strip()
        if not seed:
            continue
        text = template.replace(TEMPLATE_SEED, f"seed: {seed}")
        # 输出目录加后缀，避免多次实验互相覆盖
        base_dir = template_line(text, "directory")
        text = text.replace(base_dir, f"{base_dir}-{args.label}{seed}")
        destination = output_dir / f"{Path(args.template).stem}-{args.label}{seed}.yaml"
        destination.write_text(text, encoding="utf-8")
        print(f"生成 {destination}（seed={seed}，输出 {base_dir}-{args.label}{seed}）")


def template_line(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip()
    raise ValueError(f"模板中找不到 {key}")


if __name__ == "__main__":
    main()
