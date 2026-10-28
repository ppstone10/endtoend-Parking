"""当前闭环候选的实际全输入五层验收。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification import candidate_layer_study as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'precheck', 'capture', 'analyze', 'inspect'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--purpose', choices=('pilot', 'regression'), default='regression')
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        result = study.prepare(args.config, purpose=args.purpose, profile=args.profile)
        print({'prepared': str(args.config), 'purpose': result['purpose'], 'episodes_run': 0})
    elif args.phase == 'precheck':
        result = study.precheck(args.config)
        print({'precheck': 'passed', 'purpose': result['purpose'], 'episodes_run': 0})
    else:
        result = getattr(study, args.phase)(args.config)
        print({key: result[key] for key in ('status', 'observer_exact_to_LT26', 'elapsed_seconds',
               'layers', 'all_five_modules_qualified', 'production_qualified') if key in result})


if __name__ == '__main__':
    main()
