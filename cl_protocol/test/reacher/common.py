"""Condition definitions, receipts and paired result checks."""
import json

CASES = {
    'baseline': dict(damping=1.0, motor=1.0, rotation_deg=0.0),
    'damping_0p5': dict(damping=0.5, motor=1.0, rotation_deg=0.0),
    'damping_2': dict(damping=2.0, motor=1.0, rotation_deg=0.0),
    'damping_4': dict(damping=4.0, motor=1.0, rotation_deg=0.0),
    'motor_0p7': dict(damping=1.0, motor=0.7, rotation_deg=0.0),
    'motor_1p3': dict(damping=1.0, motor=1.3, rotation_deg=0.0),
    'rotation_30': dict(damping=1.0, motor=1.0, rotation_deg=30.0),
    'rotation_60': dict(damping=1.0, motor=1.0, rotation_deg=60.0),
}


def emit(tag, value):
    print(f'[DYNAMICS_{tag}]', json.dumps(value, ensure_ascii=False), flush=True)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def configuration(args):
    cfg = dict(policy=args.policy, dataset=args.dataset, num_eval=args.num_eval,
               seed=args.seed, cases=list(args.cases))
    if getattr(args, 'scene_manifest', None) is not None:
        cfg['scene_manifest'] = str(args.scene_manifest)
    return cfg


def validate(payload, args, case):
    if (payload.get('status') != 'complete' or payload.get('schema') != 1
            or payload.get('config') != configuration(args)
            or payload.get('condition') != CASES[case] or payload.get('case') != case
            or len(payload.get('per_scene', [])) != args.num_eval):
        raise RuntimeError(f'Existing result does not match this run: {case}')
    if abs(payload['summary']['success_rate'] - 100 * sum(
            r['success'] for r in payload['per_scene']) / args.num_eval) > 1e-7:
        raise RuntimeError(f'Invalid success counts: {case}')


def inspect_collection(path, episodes, case='damping_2'):
    import lance
    import numpy as np
    ds = lance.dataset(str(path))
    required = {'pixels', 'action', 'qpos', 'qvel', 'episode_idx', 'step_idx',
                'dynamics_damping', 'dynamics_motor_gear'}
    if not required <= set(ds.schema.names):
        raise ValueError(f'Collection schema missing {required-set(ds.schema.names)}')
    starts = ds.to_table(columns=['episode_idx'], filter='step_idx = 0').column(0).to_pylist()
    if len(set(starts)) != episodes or ds.count_rows() != episodes*201:
        raise ValueError('Expected complete 200-step episodes plus their initial frames')
    for name, expected in (('dynamics_damping', 0.01*CASES[case]['damping']),
                           ('dynamics_motor_gear', 0.05)):
        values = np.asarray(ds.to_table(columns=[name]).column(0).to_pylist())
        if not np.isfinite(values).all() or not np.allclose(values, expected):
            raise ValueError(f'Invalid recorded physical parameters: {name}')
    return dict(path=str(path), episodes=episodes, rows=ds.count_rows())


def paired_behavior(reference, candidate, cases=('baseline', 'damping_2')):
    import numpy as np
    output = []
    rng = np.random.default_rng(0)
    if len(reference) != len(cases) or len(candidate) != len(cases):
        raise ValueError('Paired behavior requires all requested conditions')
    for a, b, case in zip(reference, candidate, cases):
        signature = lambda cell: [(r['episode_idx'], r['start_step'], r['initial'])
                                  for r in cell['per_scene']]
        if signature(a) != signature(b) or a['planning'] != b['planning']:
            raise RuntimeError('Methods must use the same physical starts/goals and planning settings')
        old = np.asarray([r['success'] for r in a['per_scene']], dtype=int)
        new = np.asarray([r['success'] for r in b['per_scene']], dtype=int)
        delta = new-old
        interval = np.percentile(delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)*100,
                                 [2.5, 97.5]).tolist()
        output.append(dict(condition=case, reference=100*float(old.mean()),
                           candidate=100*float(new.mean()), change_pp=100*float(delta.mean()),
                           lost_successes=int(((old == 1) & (new == 0)).sum()),
                           gained_successes=int(((old == 0) & (new == 1)).sum()),
                           paired_bootstrap95_pp=interval))
    return output


def prediction_windows(path, count, seed):
    """Fixed windows from separately collected evaluation trajectories."""
    import lance
    import numpy as np
    ds = lance.dataset(str(path))
    table = ds.to_table(columns=['episode_idx', 'step_idx'])
    ep = np.asarray(table.column('episode_idx').to_numpy())
    step = np.asarray(table.column('step_idx').to_numpy())
    unique, inverse = np.unique(ep, return_inverse=True)
    maximum = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(maximum, inverse, step.astype(np.int64))
    valid = np.flatnonzero(step <= maximum[inverse]-15)
    chosen = np.random.default_rng(seed).choice(valid, count, replace=False)
    return [dict(episode_idx=int(ep[i]), start_step=int(step[i])) for i in chosen]
