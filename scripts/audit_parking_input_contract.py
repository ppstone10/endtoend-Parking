"""审计R0训练输入，并生成保持权重/行为不变的声明校正副本。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification.input_semantics import audit_and_correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit_and_correct(ROOT, args.output)
    print({k: v for k, v in report.items() if k != 'inputs'})


if __name__ == '__main__':
    main()
