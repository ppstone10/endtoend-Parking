"""冻结并比较参考安全过滤与完整车体约束MPC；正式任务由用户执行。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verification.candidate_study import prepare, precheck, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'precheck', 'smoke', 'run'], required=True)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--pilot', type=Path, help='正式prepare必须给出完整有收益的固定短检report.json')
    parser.add_argument('--audit', type=Path, help='正式prepare要求匹配的普通MPC独立重算报告')
    args = parser.parse_args()
    if args.phase == 'prepare':
        config = prepare(args.config, pilot=args.pilot, audit=args.audit)
        print({'prepared': str(args.config), 'purpose': config['purpose'], 'indices': len(config['indices']), 'arms': config['arms'], 'episodes_run': 0})
    elif args.phase == 'precheck':
        config = precheck(args.config)
        print({'precheck': 'passed', 'purpose': config['purpose'], 'episodes_run': 0})
    else:
        report = run(args.config, formal=args.phase == 'run')
        print({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'], 'comparisons': report['comparisons']})


if __name__ == '__main__':
    main()
