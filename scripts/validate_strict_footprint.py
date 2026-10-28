"""完整闭环严格车体约束配对：正式批量运行由用户执行。"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from verification import strict_footprint_study as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'precheck', 'run', 'inspect'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('smoke', 'retention', 'formal'), default='smoke')
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--retention-scout', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase != 'inspect' and args.config is None:
        parser.error('--config必填')
    if args.phase in ('run', 'inspect') and args.output is None:
        parser.error('--output必填')
    if args.phase == 'prepare':
        result = study.prepare(args.config, mode=args.mode, pilot=args.pilot, retention_scout=args.retention_scout)
        print({'status': 'prepared', 'tasks': len(result['indices']), 'episodes_run': 0})
    elif args.phase == 'precheck':
        result = study.precheck(args.config)
        print({'status': 'precheck_passed', 'tasks': len(result['indices']), 'episodes_run': 0})
    else:
        result = study.run(args.config, args.output) if args.phase == 'run' else study.inspect(args.output)
        print(json.dumps({'status': result['status'], 'acceptance': result['acceptance']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
