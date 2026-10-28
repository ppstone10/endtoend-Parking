"""用精确状态/标量类型修正已有v1证据，不重跑原生闭环。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.report_update import migrate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--contract-audit', type=Path)
    args = parser.parse_args()
    report = migrate_report(args.source, args.output, contract_audit=args.contract_audit)
    print({'status': report['status'], 'corrections': report['corrections'],
           'events': report['event_counts'], 'acceptance': report['acceptance']})


if __name__ == '__main__':
    main()
