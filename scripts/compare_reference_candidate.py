"""实际参考导出的余量可行控制候选：固定短检及显式人工长任务入口。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from verification.reference_study import prepare,precheck,run,inspect
from verification.reference_audit import audit
from verification.reference_teacher import run_teacher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['prepare','precheck','smoke','run','inspect','audit','teacher'])
    parser.add_argument('--config',type=Path);parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--pilot',type=Path);parser.add_argument('--audit',type=Path);parser.add_argument('--teacher',type=Path)
    args=parser.parse_args()
    if args.phase in ('prepare','precheck','smoke','run','teacher') and args.config is None:parser.error('该阶段要求--config')
    if args.phase in ('inspect','audit') and args.source is None:parser.error('该阶段要求--source')
    if args.phase=='audit' and args.output is None:parser.error('独立审计要求--output')
    if args.phase=='prepare':
        result=prepare(args.config,pilot=args.pilot,audit=args.audit,teacher=args.teacher)
        print({'prepared':str(args.config),'purpose':result['purpose'],'episodes_run':0})
    elif args.phase=='teacher':
        result=run_teacher(args.config,precheck(args.config));print({'teacher_status':result['status'],'qualified':result['qualified'],'elapsed_seconds':result['elapsed_seconds']})
    elif args.phase=='precheck':
        result=precheck(args.config);print({'precheck':'passed','purpose':result['purpose'],'episodes_run':0})
    elif args.phase in ('smoke','run'):
        result=run(args.config,formal=args.phase=='run');print({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],
                                                              'comparison':result['comparison'],'retention':result['retention'],'zero_recovery':result['zero_recovery']})
    elif args.phase=='inspect':
        result=inspect(args.source);print({'status':'completed_and_verified','candidate_qualified':result['candidate_qualified'],
                                          'production_qualified':False,'comparison':result['comparison']})
    else:
        result=audit(args.source,args.output);print({k:v for k,v in result.items() if k not in ('code','episodes')})


if __name__=='__main__':
    main()
