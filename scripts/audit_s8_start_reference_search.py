"""只读复核LT20固定协议产物；重放三个关键回合，不重新运行搜索。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.executable_behavior import PreflightBudget
from experiments.recovery_execution import record_arrays
from experiments.s8_reference_search import execution_rank, search_reference_parameters, shape_reference
from experiments.s8_clearance_surrogate import swept_penetration_summary
from interfaces import Trajectory
from scripts import search_s8_start_references as search
from scripts.prepare_recovery_balance_ab import file_digest
from training.reporting import atomic_write_json

CONFIG_SHA = '8077359c541ae34f3f9f258e6052b2998e13db189e8a42efb0736b783b49dcb8'
REPORT_SHA = 'da4f876ee9395a85b4be5c0e746bdf835a0671775bd40e6a532354742ea42af1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_console(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def equal_records(left, right, keys):
    for key in keys:
        require(np.array_equal(left[key], right[key]), f'执行记录不一致：{key}')


def audit_execution(record, summary, case, budget):
    states, commands, flags = (record[key] for key in ('states', 'commands', 'margin_violations'))
    size = summary['steps']
    require(0 < size <= 600 and states.shape == (size, 5) and commands.shape == (size, 2)
            and flags.shape == (size,) and flags.dtype == np.bool_, '执行分母/数组形状不符')
    require(np.isfinite(states).all() and np.isfinite(commands).all(), '非有限执行记录')
    previous = case['start'].to_array()[:3]
    actual_flags = []
    for state in states:
        budget.check()
        actual_flags.append(not case['planner']._collision_checker.swept_segment_free(previous, state[:3]))
        require(case['engine_kwargs']['collision_checker'].swept_segment_free(previous, state[:3]), '记录中有实体接触')
        previous = state[:3]
    require(np.array_equal(actual_flags, flags), '原检查器重算余量不一致')
    first = next((i for i, value in enumerate(flags) if value), None)
    require(summary['margin_violation_steps'] == int(flags.sum())
            and summary['first_margin_violation_step'] == first, '余量汇总不一致')
    require(summary['success'] and not summary['collision'] and not summary['truncated_preflight']
            and summary['reached_with_margin'] == (not flags.any()), '成功/严格标志不一致')
    return swept_penetration_summary(case['start'].to_array(), states, case['planner']._collision_checker, flags)


def audit_optimizer(row, directory, budget):
    attempts = row['attempts']
    cursor = 0
    generations = []

    def evaluate(parameters):
        nonlocal cursor
        require(cursor < len(attempts), 'DE重建超过登记次数')
        attempt = attempts[cursor]
        require(np.array_equal(parameters, attempt['parameters']), f'DE候选参数漂移：{cursor}')
        cursor += 1
        return attempt['rank']

    def checkpoint(generation, population, ranks, rng):
        stored = arrays(directory / f'generation-{generation:02d}.npz')
        equal_records(dict(population=population, ranks=ranks), stored, ('population', 'ranks'))
        meta = read_json(directory / f'generation-{generation:02d}.json')
        best = min(attempts[:cursor], key=lambda a: tuple(a['rank']))['number']
        require(meta == dict(generation=generation, rng_state=rng, attempts=cursor, best_attempt=best),
                '代际RNG/分母/最好引用不一致')
        generations.append(generation)

    rebuilt = search_reference_parameters(evaluate, budget, 20261023 + row['index'], on_generation=checkpoint)
    rebuilt['parameters'] = rebuilt['parameters'].tolist()
    rebuilt['score'] = list(rebuilt['score'])
    require(rebuilt == row['search_result'] and cursor == 168 and generations == list(range(6)), '搜索结果不一致')
    return dict(attempts=cursor, generations=generations, actual_objective_executions=0)


def replay(path, summary, case, seed, budget):
    stored = arrays(path)
    trajectory = Trajectory(stored['reference'], float(stored['dt']))
    result, engine, _ = search.run_case(case, None, budget, trajectory=trajectory, control_seed=seed)
    states, commands = record_arrays(result)
    equal_records(dict(states=states, commands=commands, margin_violations=np.asarray(engine.margin_violations)),
                  stored, ('states', 'commands', 'margin_violations'))
    require(search.result_row(case, result, engine) == summary, '精确重放汇总不一致')
    return dict(path=str(path.relative_to(ROOT)), control_seed=seed, steps=result.steps, exact=True)


def audit(source, output):
    source, output = source.resolve(), output.resolve()
    require(not output.is_relative_to(source), '审计输出必须在原产物目录之外')
    output.mkdir(parents=True, exist_ok=False)
    budget = PreflightBudget(180)
    started = time.perf_counter()
    torch.set_num_threads(1)
    review = dict(status='running', source=str(source), training_updates=0, search_performed=False)
    try:
        config_path, report_path = source / 'config.json', source / 'report.json'
        require(file_digest(config_path) == CONFIG_SHA and file_digest(report_path) == REPORT_SHA, 'LT20身份不符')
        config = search.checked_config(config_path, budget)
        report = read_json(report_path)
        require(report['config_sha256'] == CONFIG_SHA and report['status'] == 'completed'
                and report['stage'] == 'decision' and report['mode'] == 'formal_user_search'
                and report['complete_sources'] == 2 and report['elapsed_seconds'] < 1800, '正式完成状态不符')
        for key in ('training_updates', 'teacher_bank_created', 'main_protocol_passed', 'automatic_deployment',
                    'allowed_to_prepare_training_plan', 'qualification_passed'):
            require(report[key] == 0, f'禁止放行字段异常：{key}')
        require(report['search_performed'] is True, '原正式搜索标志不符')
        require(int(source.with_suffix('.exit-code.txt').read_text(encoding='utf-8-sig').strip()) == 0, '非零退出码')
        lines = read_console(source.with_suffix('.console.log')).splitlines()
        require(sum(line.startswith('EVAL ') for line in lines) == 170
                and sum(line.startswith('FINISHED ') for line in lines) == 1
                and 'Traceback' not in '\n'.join(lines), '控制台分母/结束状态不符')
        for name, digest in report['artifacts_sha256'].items():
            budget.check()
            require(file_digest(search.artifact_path(source, name)) == digest, f'产物摘要不符：{name}')
        review['identity'] = dict(config_sha256=CONFIG_SHA, report_sha256=REPORT_SHA,
            code_files=len(config['code']), input_files=len(config['frozen_files']), artifacts=len(report['artifacts_sha256']))
        data, definitions = search.load_sources()
        require([row['index'] for row in report['sources']] == [2338, 2351], '来源集合不符')
        episodes = transitions = 0
        review['sources'] = []
        cases = {}
        for row, (definition, case, archive, base, original) in zip(report['sources'], definitions):
            index = row['index']; cases[index] = case
            directory = source / f'source-{index}'
            attempts = row['attempts']
            require(len(attempts) == row['attempt_count'] == definition['max_attempts'], '候选分母不符')
            counts = Counter()
            for number, attempt in enumerate(attempts):
                budget.check()
                require(attempt['number'] == number and attempt['status'] == 'completed'
                        and read_json(directory / f'eval-{number:04d}/attempt.json') == attempt, '候选记录不一致')
                path = directory / f'eval-{number:04d}'
                reference = shape_reference(base, original, attempt['parameters'])
                stored = arrays(path / 'reference.npz')
                equal_records(dict(reference=reference.points, dt=np.asarray(reference.dt),
                                   parameters=np.asarray(attempt['parameters'])), stored, ('reference', 'dt', 'parameters'))
                previous = case['start'].to_array()[:3]
                unsafe = []
                for point in reference.points:
                    budget.check()
                    unsafe.append(not case['planner']._collision_checker.swept_segment_free(previous, point))
                    previous = point
                proxy = swept_penetration_summary(case['start'].to_array(), reference.points,
                                                   case['planner']._collision_checker, np.asarray(unsafe))
                require(sum(unsafe) == attempt['reference_unsafe_segments']
                        and proxy == attempt['reference_proxy'], '参考几何重算不一致')
                counts[attempt['decision']] += 1
                if 'execution' in attempt:
                    require(not any(unsafe), '不安全参考进入真实执行')
                    record = arrays(path / 'execution.npz')
                    equal_records(record, stored, ('reference', 'dt', 'parameters'))
                    proxy = audit_execution(record, attempt['execution'], case, budget)
                    require(proxy == attempt['actual_penetration_proxy'], '实际穿入代理不一致')
                    require(list(execution_rank(attempt['execution'], proxy, attempt['parameters'])) == attempt['rank'], '排名不符')
                    episodes += 1; transitions += len(record['states'])
                else:
                    require(attempt['decision'] == 'reference_geometry_rejected'
                            and any(unsafe) and not (path / 'execution.npz').exists(), '拒绝候选仍有执行记录')
                    require(attempt['rank'] == [4, sum(unsafe), proxy['max_penetration'], proxy['sum_sq_penetration'],
                            0., 0., float(np.linalg.norm(attempt['parameters']))], '几何拒绝排名不符')
            require(row['executed_episodes'] == counts['actual_mpc_evaluated'], '真实回合分母不符')
            best = min(attempts, key=lambda a: tuple(a['rank']))
            require(best['number'] == row['best_attempt'] and read_json(directory / 'best.json') == best, '最好候选引用不符')
            equal_records(arrays(directory / 'best-reference.npz'),
                          arrays(directory / f"eval-{best['number']:04d}/reference.npz"), ('reference', 'dt', 'parameters'))
            equal_records(arrays(archive), arrays(directory / 'eval-0000/execution.npz'), ('states', 'commands', 'margin_violations'))
            evidence = dict(index=index, counts=dict(counts), best_attempt=best['number'], best_execution=best['execution'])
            if index == 2338:
                evidence['optimizer'] = audit_optimizer(row, directory, budget)
            else:
                require([a['parameters'] for a in attempts] == [[0.] * 7, [-.001] * 7], '固定见证参数漂移')
                equal_records(arrays(search.SMOKE / 'source-2351/eval-0001/execution.npz'),
                              arrays(directory / 'eval-0001/execution.npz'), ('reference', 'states', 'commands', 'margin_violations'))
                verification = row['verification']
                for paired in [dict(strategy='float32-export', control_seed=0, **verification['float32_export']),
                               *verification['paired_independent_controls']]:
                    name = f"{paired['strategy']}-control{paired['control_seed']}"
                    record = arrays(directory / f'{name}.npz')
                    audit_execution(record, paired, case, budget)
                    episodes += 1; transitions += len(record['states'])
                frame = torch.load(directory / 'first-frame.pt', weights_only=False)['frame']
                # 沿用运行时局部目标核验，重新生成首窗而非相信报告clean字段。
                _, window = search.check_first_window(case, data, [frame],
                    arrays(directory / 'eval-0001/execution.npz')['states'], budget)
                require(window == verification['first_window'], '首窗重算不一致')
                require(not verification['qualified'] and not row['qualified'], '资格错误放行')
                evidence['verification'] = verification
            review['sources'].append(evidence)
        require(episodes == 135 and not any(s['qualified'] for s in report['sources']), '总分母或资格不符')
        review['episodes'] = episodes; review['transitions_rechecked'] = transitions
        best = report['sources'][0]['attempts'][129]
        paired = report['sources'][1]['verification']['paired_independent_controls']
        review['fresh_replays'] = [replay(source / 'source-2338/eval-0129/execution.npz', best['execution'], cases[2338], 0, budget)]
        for strategy in ('baseline', 'candidate'):
            summary = next(p for p in paired if p['control_seed'] == 2 and p['strategy'] == strategy)
            expected = {k: v for k, v in summary.items() if k not in ('control_seed', 'strategy')}
            review['fresh_replays'].append(replay(source / f'source-2351/{strategy}-control2.npz', expected, cases[2351], 2, budget))
        require(file_digest(report_path) == REPORT_SHA, '审计过程中原报告变化')
        review.update(status='completed', qualification_passed=False, decision='fixed_family_closed_no_teacher_bank')
    except BaseException as exc:
        review.update(status='failed', error=str(exc))
        raise
    finally:
        review['elapsed_seconds'] = time.perf_counter() - started
        review['audit_code_sha256'] = file_digest(Path(__file__))
        atomic_write_json(output / 'report.json', review)
    print(json.dumps({k: review[k] for k in ('status', 'episodes', 'transitions_rechecked', 'elapsed_seconds', 'decision')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.output)
