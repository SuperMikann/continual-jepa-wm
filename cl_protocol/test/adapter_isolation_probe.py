"""Paired adapter ablation on existing checkpoints; no training or physics.

On one updated model compare its deployed mixture, removal of new slots
without changing old weights, and renormalization over old slots only.
All three use the same encoded history, encoded target and BN state.
The original checkpoint is a reference, not a common latent-space truth.
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1]
MODES = ('deployed', 'remove_new', 'old_only')


def routed_output(x, base, mask, contributions, weights, old_count, mode, einsum):
    """Shared algebra; einsum may be torch.einsum or numpy.einsum."""
    if mode == 'deployed':
        mix = einsum('btkd,bk->btd', contributions, weights)
    elif mode in ('remove_new', 'old_only'):
        old_weights = weights[:, :old_count]
        if mode == 'old_only':
            mass = einsum('bk->b', old_weights)
            if (mass <= 0).any():
                raise ValueError('Old adapters have no gate probability')
            old_weights = old_weights / mass[:, None]
        mix = einsum('btkd,bk->btd', contributions[:, :, :old_count], old_weights)
    else:
        raise ValueError(mode)
    return x + mask * (base + mix)


class RoutingProbe:
    """Observe native intermediates; alter only the predictor's adapter mixture."""

    def __init__(self, predictor, old_count):
        self.predictor = predictor
        self.old_count = old_count
        self.mode = 'deployed'
        self.max_reconstruction_difference = 0.0
        self.new_gate_mass = []
        self.handles = [
            predictor.delta_head.register_forward_hook(self.capture),
            predictor.register_forward_hook(self.adjust),
        ]

    def capture(self, module, inputs, output):
        self.h = inputs[0]
        self.base = output

    def adjust(self, module, inputs, output):
        import torch

        h, base = self.h, self.base
        self.h = self.base = None
        x = inputs[0]
        mask, weights = module.last_mask, module.last_pi
        if x.shape != base.shape or mask.shape != base.shape:
            raise ValueError('Installed predictor has an unsupported output layout')
        if weights.shape != (x.shape[0], module.adapter_A.shape[0]):
            raise ValueError('Installed predictor has unsupported gate weights')
        u = torch.einsum('btd,krd->btkr', h, module.adapter_B)
        contributions = torch.einsum('btkr,kdr->btkd', u, module.adapter_A)
        contributions = contributions * getattr(module, 'adapter_scale', 1.0)
        rebuilt = routed_output(x, base, mask, contributions, weights,
                                self.old_count, 'deployed', torch.einsum)
        difference = float((rebuilt - output).abs().max().item())
        self.max_reconstruction_difference = max(
            self.max_reconstruction_difference, difference)
        if not torch.allclose(rebuilt, output, rtol=1e-5, atol=1e-6):
            raise ValueError(f'Native forward reconstruction failed: {difference}')
        if self.mode == 'deployed':
            self.new_gate_mass.extend(
                weights[:, self.old_count:].sum(-1).detach().cpu().tolist())
            return output
        return routed_output(x, base, mask, contributions, weights,
                             self.old_count, self.mode, torch.einsum)

    def close(self):
        for handle in self.handles:
            handle.remove()


def sample_windows(dataset, count, reserve, span, seed):
    import numpy as np
    from pred_error import episode_col

    episodes = np.asarray(dataset.get_col_data(episode_col(dataset))).reshape(-1)
    steps = np.asarray(dataset.get_col_data('step_idx')).reshape(-1)
    unique = np.unique(episodes)
    if len(unique) < reserve:
        raise ValueError(f'Only {len(unique)} episodes; requested last {reserve}')
    held = unique[-reserve:]
    valid = {}
    for episode in held:
        available = set(int(s) for s in steps[episodes == episode])
        starts = [s for s in sorted(available)
                  if all(s + offset in available for offset in range(span + 1))]
        if starts:
            valid[int(episode)] = starts
    if len(valid) < count:
        raise ValueError(f'Need {count} distinct valid episodes; found {len(valid)}')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sorted(valid), size=count, replace=False)
    scenes = [(int(ep), int(rng.choice(valid[int(ep)]))) for ep in chosen]
    return scenes, np.isin(episodes, held)


def summary(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite measurement')
    return dict(mean=float(values.mean()), std=float(values.std()), n=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-policy', default='m3g_reacher_t1/weights_epoch_30.pt')
    parser.add_argument('--updated-policy', default='m3g_reacher_t2/weights_epoch_10.pt')
    parser.add_argument('--sequence', default='B_reacher')
    parser.add_argument('--tasks', nargs='+', default=['T1', 'T2'])
    parser.add_argument('--tasks-cfg', type=Path, default=PROTOCOL / 'tasks.yaml')
    parser.add_argument('--windows', type=int, default=128,
                        help='One window per distinct episode, at most reserve-last')
    parser.add_argument('--reserve-last', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.windows, args.reserve_last, args.batch, args.stride) < 1:
        parser.error('Counts must be positive')
    if args.windows > args.reserve_last:
        parser.error('windows must not exceed reserve-last')
    if args.out.exists():
        parser.error(f'Existing result preserved: {args.out}')

    print('[ISOLATION_START] existing checkpoints; no optimizer or environment', flush=True)
    import numpy as np
    import torch
    import yaml
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm

    sys.path.insert(0, str(PROTOCOL))
    from pred_error import img_transform

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = Path(swm.wm.utils.get_cache_dir(sub_folder='checkpoints'))
    paths = [Path(p) if Path(p).is_absolute() else cache / p
             for p in (args.old_policy, args.updated_policy)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required existing checkpoint missing: {path}')
    old, updated = [swm.wm.utils.load_pretrained(str(p)).to(device).eval()
                    for p in paths]
    for model in (old, updated):
        model.requires_grad_(False)
    first, later = old.predictor, updated.predictor
    required = ('adapter_A', 'adapter_B', 'delta_head', 'mask_head')
    for predictor in (first, later):
        if not all(hasattr(predictor, name) for name in required):
            raise TypeError('Both checkpoints must use AdapterPredictor')
    old_count, total_count = first.adapter_A.shape[0], later.adapter_A.shape[0]
    if total_count <= old_count:
        raise ValueError(f'Need a grown checkpoint: old={old_count}, updated={total_count}')
    if first.adapter_A.shape[1:] != later.adapter_A.shape[1:]:
        raise ValueError('Adapter architectures differ beyond slot count')
    histories = [int(getattr(p, 'num_frames', 3)) for p in (first, later)]
    if histories[0] != histories[1]:
        raise ValueError('Checkpoints have different history lengths')
    history, span = histories[0], histories[0] * args.stride
    before = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
              for m in (old, updated)]
    slot_change = {
        name: float((getattr(later, name)[:old_count] - getattr(first, name)).abs().max())
        for name in ('adapter_A', 'adapter_B')
    }
    print('[ISOLATION_MODEL]', json.dumps({
        'old_slots': old_count, 'updated_slots': total_count,
        'freeze_gate': getattr(later, 'freeze_gate', None),
        'old_slot_max_abs_change': slot_change,
    }), flush=True)
    probe = RoutingProbe(later, old_count)
    seq = yaml.safe_load(args.tasks_cfg.read_text(encoding='utf-8'))['sequences'][args.sequence]
    task_defs = {**seq['tasks'], **seq.get('probe_tasks', {})}
    results, records = {}, {}
    transform = img_transform()
    started = time.perf_counter()
    repeat_difference = 0.0
    try:
        with torch.no_grad():
            for task in args.tasks:
                dataset = swm.data.load_dataset(task_defs[task]['dataset'],
                                               keys_to_cache=['pixels', 'action'])
                scenes, held_mask = sample_windows(
                    dataset, args.windows, args.reserve_last, span, args.seed)
                all_actions = np.asarray(dataset.get_col_data('action'))
                held_actions = all_actions[held_mask]
                scaler = StandardScaler().fit(
                    held_actions[np.isfinite(held_actions).all(axis=1)])
                rows = []
                for start in range(0, len(scenes), args.batch):
                    batch = scenes[start:start + args.batch]
                    ep = np.array([s[0] for s in batch])
                    step = np.array([s[1] for s in batch])
                    chunks = dataset.load_chunk(ep, step, step + span + 1)
                    pixels, targets, actions = [], [], []
                    for chunk in chunks:
                        pix = chunk['pixels']
                        act = np.asarray(chunk['action'])[:span]
                        if not np.isfinite(act).all():
                            raise ValueError('Non-finite recorded actions')
                        pixels.append(torch.stack([transform(pix[i * args.stride])
                                                   for i in range(history)]))
                        targets.append(transform(pix[span]))
                        actions.append(torch.from_numpy(
                            scaler.transform(act).astype(np.float32).reshape(history, -1)))
                    pixels = torch.stack(pixels).to(device)
                    targets = torch.stack(targets)[:, None].to(device)
                    actions = torch.stack(actions).to(device)
                    errors = {}
                    for model, label in ((old, 'old_reference'), (updated, 'updated')):
                        info = model.encode({'pixels': pixels.clone(), 'action': actions.clone()})
                        target = model.encode({'pixels': targets.clone()})['emb'][:, 0]
                        for mode in (('deployed',) if label == 'old_reference' else MODES):
                            probe.mode = mode
                            pred = model.predict(info['emb'].clone(), info['act_emb'].clone())[:, -1]
                            if label == 'updated' and mode == 'deployed':
                                repeat = model.predict(info['emb'].clone(), info['act_emb'].clone())[:, -1]
                                diff = float((repeat - pred).abs().max())
                                repeat_difference = max(repeat_difference, diff)
                                if not torch.allclose(repeat, pred, rtol=1e-5, atol=1e-6):
                                    raise ValueError('Native evaluation is not deterministic')
                            key = label if label == 'old_reference' else mode
                            errors[key] = (pred - target).square().mean(-1).cpu().tolist()
                    for i, (episode, window_start) in enumerate(batch):
                        rows.append(dict(episode_idx=episode, start_step=window_start,
                                         **{k: v[i] for k, v in errors.items()}))
                records[task] = rows
                task_result = {key: summary([r[key] for r in rows])
                               for key in ('old_reference', *MODES)}
                for mode in MODES[1:]:
                    difference = np.array([r[mode] - r['deployed'] for r in rows])
                    rng = np.random.default_rng(args.seed)
                    ci = np.percentile(difference[rng.integers(
                        len(rows), size=(10000, len(rows)))].mean(1), [2.5, 97.5])
                    task_result[mode + '_minus_deployed'] = summary(difference) | {
                        'bootstrap95': ci.tolist(), 'lower_error_count': int((difference < 0).sum())}
                results[task] = task_result
                print('[ISOLATION_TASK]', task, json.dumps(task_result), flush=True)
    finally:
        probe.close()
    for model, snapshot in zip((old, updated), before):
        now = model.state_dict()
        if now.keys() != snapshot.keys() or any(
                not torch.equal(now[k].detach().cpu(), v) for k, v in snapshot.items()):
            raise ValueError('Checkpoint parameters or buffers changed during evaluation')
    payload = {
        'config': vars(args) | {'out': str(args.out.resolve()), 'tasks_cfg': str(args.tasks_cfg)},
        'interpretation': 'One-step prediction ablation; known old slots, not learned routing or behavior retention',
        'sampling_note': 'One window per distinct episode from the last reserve-last episodes; historical training may have used them',
        'normalization_note': 'Action scaler fitted on last reserve-last episodes, matching pred_error.py; same scaler for all arms',
        'comparison_note': 'Updated modes share target encoding. old_reference uses its own latent coordinates and is contextual only',
        'old_adapter_count': old_count, 'updated_adapter_count': total_count,
        'freeze_gate': getattr(later, 'freeze_gate', None),
        'old_slot_max_abs_change': slot_change,
        'new_gate_mass': summary(probe.new_gate_mass),
        'native_reconstruction_max_abs_diff': probe.max_reconstruction_difference,
        'native_repeat_max_abs_diff': repeat_difference,
        'state_unchanged': True,
        'predictor_forward_source': inspect.getsource(type(later).forward),
        'results': results, 'per_window': records,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('[ISOLATION_DONE]', args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
