"""Bounded saved-state equivalence probes; never completes a parking episode."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from experiments.emitted_output_data import INDICES, case_at, load_cohort, read_json, file_digest
from interfaces import VehicleState
from scripts.diagnose_emitted_execution import PARENT
from scripts.feasible_cem_mpc import FeasibleCEMController
from scripts.runtime_refinement_mpc import RefinedCEMController
from training.reporting import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs/diagnostics/lt36-feasible-cem-20261029/results'
PROBES = ROOT / 'runs/verification/lt36-independent-review-20261029/independent-audit.json'


def probe_code():
    return {str(p.resolve()): file_digest(p) for p in (Path(__file__),
        ROOT / 'scripts/batched_footprint_check.py', ROOT / 'scripts/runtime_refinement_mpc.py',
        ROOT / 'scripts/feasible_cem_mpc.py')}


def saved_context(case, branch, step):
    path = BASELINE / f'{case["index"]:04d}-{branch}'
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as saved:
        state = VehicleState.from_array(saved['states_before'][step])
        ref = saved['control_windows'][step].copy()
        warm = np.zeros(20)
        if step:
            previous = saved['selected_control_sequences'][step-1].reshape(-1)
            warm[:-2], warm[-2:] = previous[2:], previous[-2:]
        expected = saved['selected_control_sequences'][step].reshape(-1).copy()
    rng = np.random.default_rng(case['index'])
    for _ in range(step*5):
        rng.normal(size=(80, 20))
    return state, ref, warm, expected, deepcopy(rng.bit_generator.state)


def controller(case, profile='fast', legacy=False):
    kwargs = dict(footprint_checker=case['planner']._collision_checker,
        vehicle_model=case['engine_kwargs']['vehicle_model'], dt=.1, horizon=10,
        seed=case['index'], **case['vehicle'].mpc_kwargs())
    if legacy:
        return FeasibleCEMController(**kwargs)
    return RefinedCEMController(profile=profile, goal=case['goal'],
                                terminal=case['engine_kwargs']['terminal'], **kwargs)


def run_probes(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('new probe output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    data, manifest, vehicle = load_cohort(PARENT)
    probes = read_json(PROBES)['finite_search_probes']
    if len(probes) != 19:
        raise ValueError('fixed saved-state probe denominator changed')
    rows = []
    for probe in probes:
        index, branch, step = probe['index'], probe['branch'], probe['step']
        case = case_at(data, manifest, vehicle, INDICES.index(index))
        state, ref, warm, expected, rng = saved_context(case, branch, step)
        old, fast = controller(case, legacy=True), controller(case)
        old._rng.bit_generator.state, fast._rng.bit_generator.state = deepcopy(rng), deepcopy(rng)
        timings, sequences = [], []
        for c in (old, fast):
            begin = time.perf_counter()
            sequences.append(c._solve(warm, ref, state))
            timings.append(1000.*(time.perf_counter()-begin))
        np.testing.assert_array_equal(sequences[0], expected)
        np.testing.assert_array_equal(sequences[0], sequences[1])
        for a, b in zip(old.safety_audit()['generations'], fast.safety_audit()['generations']):
            for key in ('feasible_mask', 'elite_indices'):
                if a[key] != b[key]:
                    raise ValueError(f'pure acceleration changed {key}: {index}/{branch}/{step}')
        if old._rng.bit_generator.state != fast._rng.bit_generator.state:
            raise ValueError('pure acceleration changed random stream')
        rows.append(dict(index=index, branch=branch, step=step, exact=True,
                         scalar_ms=timings[0], batch_ms=timings[1]))
    # Exercise the specific sparse-elite and terminal-certificate witnesses.
    case = case_at(data, manifest, vehicle, INDICES.index(860))
    state, ref, warm, _, rng = saved_context(case, 'teacher_fixed', 0)
    guarded = controller(case, 'variance')
    guarded._rng.bit_generator.state = rng
    guarded._solve(warm, ref, state)
    first = guarded.safety_audit()['generations'][0]
    if first['feasible_samples'] != 1 or not first['sparse_elite_guard']:
        raise ValueError('real single-elite witness not guarded')
    np.testing.assert_array_equal(first['std_after'], first['std_before'])
    case = case_at(data, manifest, vehicle, INDICES.index(1290))
    terminal = controller(case, 'terminal')
    with np.load(BASELINE / '1290-network_rolling.npz', allow_pickle=False) as saved:
        safe_pose = VehicleState.from_array(saved['states_after'][537])
        unsafe_pose = VehicleState.from_array(saved['states_after'][569])
    yes, yes_audit = terminal._terminal_sequence(safe_pose)
    no, no_audit = terminal._terminal_sequence(unsafe_pose)
    if yes is None or no is not None or not terminal.sequence_safe(yes, safe_pose):
        raise ValueError('real full-rotation margin certificates changed')
    report = dict(status='passed', code=probe_code(), scope='19 saved CEM calls; no full episode or recovery claim',
        rows=rows, exact_probes=len(rows), sparse_elite_witness=first,
        terminal_certificate_pass=yes_audit, terminal_certificate_reject=no_audit,
        scalar_p50_ms=float(np.median([r['scalar_ms'] for r in rows])),
        batch_p50_ms=float(np.median([r['batch_ms'] for r in rows])),
        batch_max_ms=max(r['batch_ms'] for r in rows),
        candidate_qualified=False, training_performed=False)
    atomic_write_json(output, report)
    return report
