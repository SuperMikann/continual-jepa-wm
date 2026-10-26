"""Compare old/current planner scores on identical old-task inputs and plans.

This is an offline teacher/student diagnostic, not a physical rollout or a
success-rate evaluation. It supports PushT because it does not restore states.
Candidates follow the initial Gaussian shooting population; no CEM iterations
are run. No training or checkpoint modification is performed.

The last --holdout episodes form the probe pool. This does not establish that
these episodes were excluded from the checkpoints' historical training.
"""

import argparse
import json
import math
import os
from pathlib import Path


def kendall_tau(x, y):
    """Tau-b; undefined for a constant score vector."""
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            ties_x += dx == 0
            ties_y += dy == 0
            concordant += dx * dy > 0
            discordant += dx * dy < 0
    pairs = len(x) * (len(x) - 1) / 2
    denominator = math.sqrt((pairs - ties_x) * (pairs - ties_y))
    return (concordant - discordant) / denominator if denominator else None


def standardize(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return [(x - mean) / std for x in values] if std else [0.0] * len(values), std


def log_preferences(scores, temperature):
    logits = [-x / temperature for x in scores]
    peak = max(logits)
    normalizer = peak + math.log(sum(math.exp(x - peak) for x in logits))
    return [x - normalizer for x in logits]


def score_metrics(teacher, student, topk, temperature):
    if len(teacher) != len(student) or len(teacher) < 2:
        raise ValueError('Need two equally sized candidate cost vectors.')
    if not all(math.isfinite(x) for x in teacher + student):
        raise ValueError('Non-finite planner costs; inspect model/inputs first.')
    if not 1 <= topk <= len(teacher) or temperature <= 0:
        raise ValueError('Invalid topk or temperature.')
    t_order = sorted(range(len(teacher)), key=lambda i: teacher[i])
    s_order = sorted(range(len(student)), key=lambda i: student[i])
    t_norm, t_std = standardize(teacher)
    s_norm, s_std = standardize(student)
    t_logp = log_preferences(t_norm, temperature)
    s_logp = log_preferences(s_norm, temperature)
    kl = sum(math.exp(a) * (a - b) for a, b in zip(t_logp, s_logp))
    teacher_best = teacher[t_order[0]]
    teacher_worst = teacher[t_order[-1]]
    best_ties = sum(x == teacher_best for x in teacher)
    return {
        'teacher_student_tau': kendall_tau(teacher, student),
        'top1_agreement': float(teacher[s_order[0]] == teacher_best),
        'topk_overlap': len(set(t_order[:topk]) & set(s_order[:topk])) / topk,
        'teacher_regret_std': (
            (teacher[s_order[0]] - teacher_best) / t_std if t_std else None
        ),
        'preference_kl': max(0.0, kl),
        'teacher_cost_mean': sum(teacher) / len(teacher),
        'student_cost_mean': sum(student) / len(student),
        'teacher_cost_std': t_std,
        'student_cost_std': s_std,
        'teacher_best_ties': best_ties,
        'teacher_cost_range': teacher_worst - teacher_best,
    }


def aggregate(records, key):
    values = [r[key] for r in records if r[key] is not None]
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    mean = sum(values) / len(values)
    return {
        'mean': mean,
        'std': math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)),
        'n': len(values),
    }


def self_test():
    costs = [0.0, 1.0, 2.0, 4.0]
    identical = score_metrics(costs, costs, 2, 1.0)
    affine = score_metrics(costs, [3 * x + 10 for x in costs], 2, 1.0)
    reverse = score_metrics(costs, [-x for x in costs], 2, 1.0)
    constant = score_metrics([1.0] * 4, costs, 2, 1.0)
    assert identical['teacher_student_tau'] == 1
    assert identical['top1_agreement'] == 1
    assert identical['teacher_regret_std'] == 0
    assert identical['preference_kl'] < 1e-12
    assert affine['teacher_student_tau'] == 1
    assert affine['preference_kl'] < 1e-12
    assert reverse['teacher_student_tau'] == -1
    assert reverse['top1_agreement'] == 0
    assert reverse['teacher_regret_std'] > 0
    assert constant['teacher_student_tau'] is None
    assert constant['teacher_regret_std'] is None
    assert kendall_tau([0, 0, 1], [0, 1, 1]) == 0.5
    json.dumps(constant, allow_nan=False)
    print('[SELF_TEST_OK] identical, affine scaling, reverse ranking, ties, flat costs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--tasks', default=str(Path(__file__).with_name('tasks.yaml')))
    parser.add_argument('--sequence', default='A_pusht')
    parser.add_argument('--task', default='T1')
    parser.add_argument('--teacher-policy')
    parser.add_argument('--student-policy')
    parser.add_argument('--dataset', help='Optional common source dataset override.')
    parser.add_argument('--num-scenes', type=int, default=20)
    parser.add_argument('--num-candidates', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--goal-offset', type=int, default=25)
    parser.add_argument('--holdout', type=int, default=200)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='One scene, four candidates.')
    parser.add_argument('--out', default='planning_retention_probe.json')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.teacher_policy or not args.student_policy:
        parser.error('--teacher-policy and --student-policy are required.')
    if args.check:
        args.num_scenes, args.num_candidates = 1, 4
    args.topk = min(args.topk, args.num_candidates)
    if min(args.num_scenes, args.horizon, args.frameskip, args.holdout) < 1:
        parser.error('Scene count, horizon, frameskip and holdout must be positive.')
    if args.num_candidates < 2 or args.topk < 1 or args.temperature <= 0:
        parser.error('Need at least two candidates, positive topk and temperature.')

    # Fail fast on missing local checkpoints before importing the ML stack.
    cache = os.environ.get('STABLEWM_HOME')
    if cache:
        for policy in (args.teacher_policy, args.student_policy):
            path = Path(policy)
            path = path if path.is_absolute() else Path(cache) / 'checkpoints' / path
            if not path.is_file():
                raise FileNotFoundError(f'Checkpoint missing: {path}')

    import numpy as np
    import torch
    import yaml
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from pred_error import episode_col, img_transform

    with open(args.tasks, encoding='utf-8') as file:
        sequence = yaml.safe_load(file)['sequences'][args.sequence]
    task = {**sequence['tasks'], **sequence.get('probe_tasks', {})}[args.task]
    dataset_name = args.dataset or task['dataset']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = swm.data.load_dataset(dataset_name, keys_to_cache=['pixels', 'action'])
    episodes = np.asarray(dataset.get_col_data(episode_col(dataset)))
    steps = np.asarray(dataset.get_col_data('step_idx'))
    unique_episodes = np.unique(episodes)
    pool_episodes = unique_episodes[-args.holdout:]
    actions = np.asarray(dataset.get_col_data('action'))
    if actions.ndim != 2:
        raise ValueError(f'Expected action column (N,A), got {actions.shape}.')
    finite_actions = actions[np.isfinite(actions).all(axis=1)]
    scaler = StandardScaler().fit(finite_actions)
    action_dim = actions.shape[1]
    candidate_dim = action_dim * args.frameskip

    teacher = swm.wm.utils.load_pretrained(args.teacher_policy).to(device).eval()
    student = swm.wm.utils.load_pretrained(args.student_policy).to(device).eval()
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    history = int(getattr(teacher.predictor, 'num_frames', 3))
    if int(getattr(student.predictor, 'num_frames', 3)) != history:
        raise ValueError('Teacher/student history lengths differ.')
    if args.goal_offset <= (history - 1) * args.frameskip:
        raise ValueError('Goal frame must be after the history window.')
    teacher_eval = ShootingCostEvaluator(teacher, GoalMSE())
    student_eval = ShootingCostEvaluator(student, GoalMSE())
    span = max(args.goal_offset, history * args.frameskip - 1)
    valid = []
    for episode in pool_episodes:
        episode_steps = np.unique(steps[episodes == episode])
        maximum = int(episode_steps.max()) - span
        valid.extend((int(episode), int(s)) for s in episode_steps if s <= maximum)
    if len(valid) < args.num_scenes:
        raise ValueError(f'Only {len(valid)} valid windows for {args.num_scenes} scenes.')
    rng = np.random.default_rng(args.seed)
    scenes = [valid[i] for i in rng.choice(len(valid), args.num_scenes, replace=False)]
    generator = torch.Generator().manual_seed(args.seed)
    candidates = torch.randn(
        1, args.num_candidates, args.horizon, candidate_dim, generator=generator
    ).to(device)
    transform = img_transform()
    records = []
    print(f'[CONFIG] {args.sequence}/{args.task}; dataset={dataset_name}; device={device}', flush=True)
    print(f'[PAIR] teacher={args.teacher_policy}; student={args.student_policy}', flush=True)
    print(f'[POOL] scenes={args.num_scenes}; candidates={args.num_candidates}; history={history}', flush=True)

    for index, (episode, start) in enumerate(scenes):
        chunk = dataset.load_chunk(
            np.array([episode]), np.array([start]), np.array([start + span + 1])
        )[0]
        pixels = chunk['pixels']
        window = torch.stack([transform(pixels[i * args.frameskip]) for i in range(history)])
        goal = transform(pixels[args.goal_offset])
        raw_history = np.asarray(chunk['action'][:history * args.frameskip])
        if not np.isfinite(raw_history).all():
            raise ValueError(f'Non-finite history actions at episode={episode}, start={start}.')
        historical_actions = scaler.transform(raw_history).astype(np.float32).reshape(history, candidate_dim)

        def costs(evaluator):
            # Each call owns its dictionary and tensors: encode/evaluator may mutate them.
            count = args.num_candidates
            info = {
                'pixels': window[None, None].expand(1, count, history, *window.shape[1:]).clone().to(device),
                'goal': goal[None, None, None].expand(1, count, 1, *goal.shape).clone().to(device),
                'action': torch.from_numpy(historical_actions)[None, None]
                    .expand(1, count, history, candidate_dim).clone().to(device),
            }
            with torch.no_grad():
                result = evaluator.get_cost(info, candidates.clone())[0].detach().cpu().reshape(-1)
            if result.numel() != count:
                raise ValueError(f'Expected {count} candidate costs, got {tuple(result.shape)}.')
            return [float(x) for x in result.tolist()]

        teacher_costs, student_costs = costs(teacher_eval), costs(student_eval)
        metrics = score_metrics(teacher_costs, student_costs, args.topk, args.temperature)
        record = {
            'scene': index, 'ep': episode, 'start': start,
            **metrics, 'teacher_costs': teacher_costs, 'student_costs': student_costs,
        }
        records.append(record)
        tau = metrics['teacher_student_tau']
        tau_text = f'{tau:+.3f}' if tau is not None else 'undefined'
        regret = metrics['teacher_regret_std']
        regret_text = f'{regret:.3f}' if regret is not None else 'undefined'
        print(
            f'[SCENE {index + 1}/{args.num_scenes}] ep={episode} start={start} '
            f'tau_teacher_student={tau_text} top1={metrics["top1_agreement"]:.0f} '
            f'top{args.topk}={metrics["topk_overlap"]:.3f} '
            f'teacher_regret_std={regret_text} KL={metrics["preference_kl"]:.4f}',
            flush=True,
        )

    metric_keys = list(score_metrics([0.0, 1.0], [0.0, 1.0], 1, 1.0))
    summary = {key: aggregate(records, key) for key in metric_keys}
    output = {
        'config': vars(args), 'dataset': dataset_name,
        'metric_reference': 'teacher planner scores, not physical outcomes',
        'candidate_source': 'initial Gaussian population; no CEM refinement',
        'historical_training_exposure': 'unknown; last episodes are a probe pool only',
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'candidate_actions_standardized': candidates.cpu().tolist(),
        'summary': summary, 'per_scene': records,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('[SUMMARY]', flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print(f'[DONE] saved={out}', flush=True)


if __name__ == '__main__':
    main()
