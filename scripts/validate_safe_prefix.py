"""安全前缀正式全量交接；prepare/precheck零回合。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from verification import safe_prefix_handoff as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'precheck', 'run', 'inspect'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase in ('prepare', 'precheck', 'run') and args.config is None:
        parser.error('--config必填')
    if args.phase == 'prepare' and args.pilot is None:
        parser.error('--pilot必填')
    if args.phase in ('run', 'inspect') and args.output is None:
        parser.error('--output必填')
    if args.phase == 'prepare':
        study.prepare(args.config, args.pilot)
        print({'status': 'prepared', 'episodes_run': 0})
    elif args.phase == 'precheck':
        study.precheck(args.config)
        print({'status': 'precheck_passed', 'episodes_run': 0})
    else:
        result = study.run(args.config, args.output) if args.phase == 'run' else study.inspect(args.output)
        print({k: result[k] for k in ('status', 'successes', 'lost_lt32_successes',
            'lost_native_successes_in_evaluated_set', 'lost_historical_successes_in_evaluated_set', 'candidate_qualified')})


if __name__ == '__main__':
    main()
