"""只读核验候选产物、完整分母及联合裁决，不运行控制或训练。"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification.candidate_study import precheck, read, native_rows
from verification.candidate_execution import compare_rows
from scripts.diagnose_closed_loop_failures import _sha256


def inspect(source):
    source = Path(source).resolve()
    config = precheck(source / 'config.json')
    report = read(source / 'report.json')
    if report.get('status') != 'completed' or report['indices'] != config['indices']:
        raise ValueError('运行未完整完成或分母不符')
    artifacts = read(source / 'artifacts.json')
    required = {'report.json', 'config.json', 'summary.md'}
    for arm in config['arms']:
        required.add(f'{arm}/raw-report.json')
        required.update(f'{arm}/episode-{i:04d}.{suffix}' for i in config['indices'] for suffix in ('json', 'npz'))
    if not required <= set(artifacts):
        raise ValueError('候选产物登记不完整')
    for name, digest in artifacts.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or _sha256(path) != digest:
            raise ValueError('候选产物缺失或漂移')
    if _sha256(source / 'config.json') != report['identity']['config_sha256']:
        raise ValueError('报告配置SHA不符')
    if config['purpose'] == 'formal' and report['rows']['native'] != native_rows(config['source']):
        raise ValueError('缓存基线与已验收LT21不一致')
    qualified = False
    for arm in config['arms']:
        rows = report['rows'][arm]
        if [r['dataset_index'] for r in rows] != config['indices']:
            raise ValueError('回合索引缺失/顺序漂移')
        for row in rows:
            episode = read(source / arm / f'episode-{row["dataset_index"]:04d}.json')
            if episode['summary'] != row or len(episode['frames']) != row['steps']:
                raise ValueError('逐回合账本不完整')
            if arm in ('control', 'combined') and not all(f['selected_horizon_safe'] and f['actual_contact_safe'] for f in episode['frames']):
                raise ValueError('候选安全约束在已完成账本中失效')
        if arm == 'native':
            continue
        paired = compare_rows(report['rows']['native'], rows)
        paired['formal_joint_pass'] = bool(config['purpose'] == 'formal' and paired['samples'] == 300
                                          and paired['successes'] >= 210 and paired['collisions'] <= 30
                                          and not paired['old_success_lost'])
        if paired != report['comparisons'][arm]:
            raise ValueError('配对汇总与联合裁决不一致')
        qualified |= paired['formal_joint_pass']
    if qualified != report['candidate_qualified']:
        raise ValueError('总体资格不一致')
    return {'status': 'completed_and_verified', 'mode': report['mode'],
            'candidate_qualified': qualified, 'comparisons': report['comparisons'],
            'geometry_source': 'simulator_ground_truth', 'production_qualified': False,
            'scope': '完整场景几何辅助的系统实验；BEV占用接入控制约束尚未实现，指标门槛不能替代生产准入。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    print(inspect(args.source))


if __name__ == '__main__':
    main()
