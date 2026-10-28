"""版本化修复旧image数据的相机目标通道，不启动训练或闭环。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification.surround_rebuild import repair_camera_targets, inspect_camera_target_repair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='必须为不存在的新目录')
    parser.add_argument('--limit', type=int, default=0, help='0为完整；正数仅前缀预检')
    parser.add_argument('--budget-seconds', type=float, default=3600)
    parser.add_argument('--manifest', type=Path, help='合并训练源无manifest时，显式提供原任务manifest.json')
    parser.add_argument('--original-data', type=Path, help='原任务归档，用于严格认证恢复机动标签改写的谱系')
    parser.add_argument('--inspect', action='store_true', help='只核验已完成副本及源身份')
    args = parser.parse_args()
    if args.inspect:
        print(inspect_camera_target_repair(args.output))
        return
    if args.data is None:
        parser.error('重建需要--data')
    report = repair_camera_targets(args.data, args.output, limit=args.limit, budget_seconds=args.budget_seconds,
                                   manifest_path=args.manifest, original_data_path=args.original_data)
    print({k: report[k] for k in ('status', 'rebuilt', 'partial', 'output_data', 'elapsed_seconds')})


if __name__ == '__main__':
    main()
