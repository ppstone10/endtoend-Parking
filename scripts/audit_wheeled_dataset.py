"""轻量只读轮式必要条件审核；不加载BEV、不生成新标签。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.wheeled_audit import audit_archive
from sim.wheeled_config import load_wheeled_vehicle_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--config", default="configs/vehicles/wheeled_research_v1.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_wheeled_vehicle_config(args.config)
    reports = [audit_archive(path, config) for path in args.data]
    if args.output:
        encoded = json.dumps(dict(kind="wheeled_necessary_conditions", reports=reports),
                             ensure_ascii=False, indent=2, allow_nan=False)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(json.dumps([{key: value for key, value in report.items() if key != "rows"}
                      for report in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
