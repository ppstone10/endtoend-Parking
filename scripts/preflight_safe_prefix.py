"""安全前缀固定短检；不支持自动全量运行。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from verification.safe_prefix_preflight import run, select_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('regressions', 'expanded'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if args.list_only:
        print({'indices': select_indices(args.stage), 'episodes_run': 0})
        return
    if args.output is None:
        parser.error('--output必填')
    torch.set_num_threads(1)
    result = run(args.stage, args.output)
    print({key: result[key] for key in ('status', 'successes', 'lost_lt32_successes', 'gained_successes', 'all_three_regressions_restored')})


if __name__ == '__main__':
    main()
