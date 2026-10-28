"""独立核验固定四回合的候选选择/成本/窗口/实际状态与完整车体安全。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.candidate_audit import audit_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit_control(args.source, args.output)
    print({k: v for k, v in report.items() if k != 'code'})


if __name__ == '__main__':
    main()
