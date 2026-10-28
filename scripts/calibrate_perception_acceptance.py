"""训练源感知经验参考校准；预检结果不可作为质量通过证据。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.candidate_layer_study import prepare, precheck
from verification.perception_calibration import run_calibration, load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'precheck', 'run', 'inspect'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.phase in ('run', 'inspect') and args.output is None:
        parser.error('run/inspect要求--output')
    if args.phase == 'prepare':
        result = prepare(args.config)
        print({'prepared': str(args.config), 'episodes_run': 0})
    elif args.phase == 'precheck':
        precheck(args.config);print({'precheck': 'passed', 'episodes_run': 0})
    elif args.phase == 'run':
        print(run_calibration(args.config, args.output, preflight=args.preflight))
    else:
        profile = load_profile(args.output, precheck(args.config))
        print({'status': profile['status'], 'preflight': profile['preflight'],
               'groups': len(profile['groups']), 'absolute_quality_status': profile['absolute_quality_status']})


if __name__ == '__main__':
    main()
