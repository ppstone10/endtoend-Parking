"""显式终端转向实验：固定短检由Agent预检，正式同300由用户运行。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification.terminal_study import prepare, precheck, run, inspect
from verification.terminal_audit import audit
from verification.terminal_diagnosis import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['prepare','precheck','smoke','run','inspect','audit','diagnose'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.phase in ('prepare','precheck','smoke','run') and args.config is None:
        parser.error('该阶段要求--config')
    if args.phase in ('inspect','audit') and args.source is None:
        parser.error('该阶段要求--source')
    if args.phase in ('audit','diagnose') and args.output is None:
        parser.error('该阶段要求--output')
    if args.phase == 'prepare':
        result = prepare(args.config, pilot=args.pilot, audit=args.audit)
        print({'prepared':str(args.config), 'purpose':result['purpose'], 'episodes_run':0})
    elif args.phase == 'precheck':
        result = precheck(args.config); print({'precheck':'passed','purpose':result['purpose'],'episodes_run':0})
    elif args.phase in ('smoke','run'):
        result = run(args.config, formal=args.phase=='run')
        print({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'comparison':result['comparison']})
    elif args.phase == 'inspect':
        result = inspect(args.source)
        print({'status':'completed_and_verified','mode':result['mode'],'candidate_qualified':result['candidate_qualified'],
               'production_qualified':False,'comparison':result['comparison']})
    elif args.phase == 'audit':
        result = audit(args.source,args.output)
        print({k:v for k,v in result.items() if k not in ('episodes','code')})
    else:
        result = diagnose(args.output)
        print({k:v for k,v in result.items() if k!='episodes'})


if __name__ == '__main__':
    main()
