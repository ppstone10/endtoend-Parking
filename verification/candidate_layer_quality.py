"""五单元质量规则；不以重放一致覆盖危险输出或未到达。"""
LAYERS = ('L1', 'L2', 'L3', 'L1_L2', 'L2_L3')


def layer_verdict(evidence, errors, failures=(), blocked=(), *, absolute=True):
    failures, blocked = list(failures), list(blocked)
    if not evidence:
        blocked.append('missing_evidence')
    quality = 'failed' if failures else 'blocked' if blocked else 'passed'
    qualified = bool(evidence and not errors and quality == 'passed')
    return {'evidence_status': 'complete' if evidence else 'incomplete',
            'contract_status': 'failed' if errors else 'passed' if evidence else 'blocked',
            'quality_status': quality, 'quality_failures': failures, 'quality_blockers': blocked,
            'contract_errors': list(errors), 'covered_scope_qualified': qualified,
            'absolute_quality_status': 'covered_scope_only' if absolute else 'not_established',
            'module_qualified': bool(qualified and absolute)}


def episode_verdict(facts, errors, profile_results):
    """facts是独立几何/任务观察，不接受调用者传入通过标志。"""
    evidence = facts['inputs_complete']
    reference_failures = []
    if facts['physical_unsafe_references']:
        reference_failures.append('unsafe_full_reference_contact')
    if facts['margin_unsafe_references']:
        reference_failures.append('unsafe_full_reference_margin')
    if facts['near_goal_unsupported']:
        reference_failures.append('near_goal_without_joint_support')
    reference_blockers = [] if facts['near_goal_queries'] else ['terminal_support_not_observed']
    perception_failures = ['outside_perception_reference'] if 'failed' in profile_results else []
    perception_blockers = ['perception_reference_unavailable'] if not profile_results or 'blocked' in profile_results else []
    control_failures = []
    if facts['unsafe_actual_steps']:
        control_failures.append('unsafe_actual_transition')
    if facts['unsafe_horizons']:
        control_failures.append('unsafe_selected_horizon')
    task_failures = [] if facts['success'] else ['original_task_not_reached']
    result = {
        'L1': layer_verdict(evidence, errors['L1'], perception_failures, perception_blockers, absolute=False),
        'L2': layer_verdict(evidence, errors['L2'], reference_failures, reference_blockers),
        'L3': layer_verdict(evidence, errors['L3'], control_failures),
        'L1_L2': layer_verdict(evidence, errors['L1_L2'], perception_failures + reference_failures,
                             perception_blockers + reference_blockers, absolute=False),
        'L2_L3': layer_verdict(evidence, errors['L2_L3'], control_failures + task_failures)}
    return result


def aggregate_verdicts(rows):
    if not rows:
        raise ValueError('空分母不能形成五层验收')
    result = {}
    for name in LAYERS:
        values = [row['layers'][name] for row in rows]
        result[name] = {
            'evidence_status': 'complete' if all(v['evidence_status'] == 'complete' for v in values) else 'incomplete',
            'contract_status': 'failed' if any(v['contract_status'] == 'failed' for v in values) else
                               'blocked' if any(v['contract_status'] == 'blocked' for v in values) else 'passed',
            'quality_status': 'failed' if any(v['quality_status'] == 'failed' for v in values) else
                              'blocked' if any(v['quality_status'] == 'blocked' for v in values) else 'passed',
            'failed_indices': [r['dataset_index'] for r in rows if r['layers'][name]['quality_status'] == 'failed'],
            'blocked_indices': [r['dataset_index'] for r in rows if r['layers'][name]['quality_status'] == 'blocked'],
            'covered_scope_qualified': all(v['covered_scope_qualified'] for v in values),
            'module_qualified': all(v['module_qualified'] for v in values)}
    return result
