"""Read-only P1 follow-up: stored vs fresh plans on the same memory states.

No training, environment rollout, checkpoint modification, or solver replacement.
Also saves the installed planning sources/configuration for deployment inspection.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path


def source_inventory(root):
    """Read the installed implementation, rather than assume upstream defaults."""
    spec = importlib.util.find_spec('stable_worldmodel')
    package = Path(spec.origin).parent
    entry = root / 'scripts' / 'plan' / 'eval_wm.py'
    if not entry.is_file():
        raise FileNotFoundError(entry)
    sources = {}
    for path in package.rglob('*.py'):
        relative = str(path.relative_to(package))
        if 'planning' in path.relative_to(package).parts or any(
            word in path.stem.lower() for word in ('actor', 'planner', 'policy')
        ) or path.name in ('lewm.py', 'world.py'):
            sources[relative] = path.read_text(encoding='utf-8')
    configs = {}
    entry_text = entry.read_text(encoding='utf-8')
    declared = []
    for node in ast.walk(ast.parse(entry_text)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == 'config_path' and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    if isinstance(value, str):
                        declared.append((entry.parent / value).resolve())
    available = [path for path in declared if path.is_dir()]
    if not available:
        available = [
            path for path in (entry.parent / 'config', entry.parent / 'configs')
            if path.is_dir()
        ]
    if len(set(available)) != 1:
        raise ValueError(f'Cannot identify the plan config directory: {available}')
    config_root = available[0]
    for path in sorted(config_root.rglob('*')):
        if path.suffix in ('.yaml', '.yml'):
            configs[str(path.relative_to(config_root))] = path.read_text(encoding='utf-8')
    constructors = {}
    for relative, source in sources.items():
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and (
                'cem' in node.name.lower() or 'crossentropy' in node.name.lower()
            ):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == '__init__':
                        constructors[relative + ':' + node.name] = '\n'.join(
                            lines[child.lineno - 1:child.end_lineno]
                        )
    return {
        'package': str(package), 'eval_entrypoint_path': str(entry),
        'eval_entrypoint': entry.read_text(encoding='utf-8'),
        'config_root': str(config_root),
        'plan_configs': configs, 'planning_sources': sources,
        'cem_constructors': constructors,
        'note': 'Source configuration only; resolved runtime overrides are not inferred.',
    }


def to_list(costs, count):
    import torch
    if tuple(costs.shape) != (1, count) or not torch.isfinite(costs).all():
        raise ValueError(f'Invalid cost tensor: {tuple(costs.shape)}')
    return [float(value) for value in costs[0].detach().cpu().tolist()]


def metrics(old, new, topk):
    from planning_retention_probe import score_metrics
    values = score_metrics(old, new, topk, 1.0)
    names = {
        'teacher_student_tau': 'old_new_tau',
        'top1_agreement': 'best_choice_agreement',
        'topk_overlap': 'topk_overlap',
        'teacher_regret_std': 'old_score_gap_std',
        'preference_kl': 'normalized_score_kl',
    }
    return {target: values[source] for source, target in names.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/root/autodl-tmp/stable-worldmodel')
    parser.add_argument('--memory', default='results/p1_pusht_memory_seed0.pt')
    parser.add_argument('--out', default='results/p1_memory_coverage_audit.json')
    parser.add_argument('--old-policy', default='pusht_seed/weights_epoch_30.pt')
    parser.add_argument('--new-policies', nargs='+', default=[
        'p1_pusht_finetune_s0_300/weights_step_300.pt',
        'p1_pusht_replay_s0_300/weights_step_300.pt',
        'p1_pusht_score_kd_s0_300/weights_step_300.pt',
    ])
    parser.add_argument('--fresh-seed', type=int, default=20261010)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    memory_path = Path(args.memory)
    memory_path = memory_path if memory_path.is_absolute() else root / memory_path
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    source_out = out.with_name(out.stem + '_sources.json')
    for path in (out, source_out):
        if path.exists():
            raise FileExistsError(f'Keep previous output; choose another --out: {path}')
    if not memory_path.is_file():
        raise FileNotFoundError(memory_path)
    native = source_inventory(root)
    source_out.parent.mkdir(parents=True, exist_ok=True)
    source_out.write_text(json.dumps(native, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[SOURCE_SAVED]', source_out, flush=True)

    import torch
    import stable_worldmodel as swm
    from planning_retention_probe import aggregate
    from train_planning_retention import load_memory_batch, score_plans

    if not torch.cuda.is_available():
        raise RuntimeError('Run this model check on the AutoDL GPU.')
    device = 'cuda'
    bank = torch.load(memory_path, map_location='cpu', weights_only=False)
    if bank['manifest']['teacher'] != args.old_policy:
        raise ValueError('Old checkpoint must match the memory label source.')
    stored = bank['candidates']
    if stored.ndim != 4 or stored.shape[1] < 2 or len(stored) != len(bank['pixels']):
        raise ValueError('Expected memory candidates (N,S,T,A) and aligned pixels.')
    if not torch.isfinite(stored).all() or not torch.isfinite(bank['teacher_costs']).all():
        raise ValueError('Non-finite saved plans or old-model labels.')
    count = stored.shape[1]
    topk = min(2, count)
    generator = torch.Generator().manual_seed(args.fresh_seed)
    fresh = torch.randn(stored.shape, generator=generator, dtype=stored.dtype)
    policies = [args.old_policy] + args.new_policies
    if len(set(policies)) != len(policies):
        raise ValueError('Checkpoint list contains duplicates.')
    cache = Path(os.environ['STABLEWM_HOME']) / 'checkpoints'
    for policy in policies:
        path = Path(policy)
        path = path if path.is_absolute() else cache / path
        if not path.is_file():
            raise FileNotFoundError(path)

    reference = {'stored': [], 'fresh': []}
    per_scene = []
    summaries = {}
    reproduction = {}
    for policy_index, policy in enumerate(policies):
        print('[MODEL]', policy, flush=True)
        model = swm.wm.utils.load_pretrained(policy).to(device).eval()
        model.requires_grad_(False)
        if int(getattr(model.predictor, 'num_frames', 3)) != 3:
            raise ValueError('This audit reuses the P1 history=3 protocol.')
        rows = {'stored': [], 'fresh': []}
        maximum_label_difference = 0.0
        with torch.inference_mode():
            for i in range(len(stored)):
                batch = load_memory_batch(bank, [i], device)
                arguments = (
                    batch['pixels'][:, :3], batch['goal'],
                    batch['candidates'], batch['action'][:, :3],
                )
                stored_cost = score_plans(model, *arguments, differentiable=False)
                fresh_cost = score_plans(
                    model, batch['pixels'][:, :3], batch['goal'],
                    fresh[i:i + 1].to(device), batch['action'][:, :3],
                    differentiable=False,
                )
                costs = {
                    'stored': to_list(stored_cost, count),
                    'fresh': to_list(fresh_cost, count),
                }
                if policy_index == 0:
                    saved = bank['teacher_costs'][i:i + 1].to(device)
                    maximum_label_difference = max(
                        maximum_label_difference,
                        float((stored_cost - saved).abs().max()),
                    )
                    torch.testing.assert_close(
                        stored_cost, saved, rtol=1e-4, atol=1e-3,
                        msg=f'Old-model labels cannot be reproduced for memory scene {i}.',
                    )
                    for population in costs:
                        reference[population].append(costs[population])
                for population in costs:
                    values = metrics(reference[population][i], costs[population], topk)
                    rows[population].append(values)
                    per_scene.append({
                        'policy': policy, 'memory_index': i,
                        'ep_start': bank['manifest']['scenes'][i],
                        'population': population, **values,
                        'old_costs': reference[population][i],
                        'new_costs': costs[population],
                    })
                if (i + 1) % 16 == 0:
                    print(f'[MEMORY {i + 1}/{len(stored)}] {policy}', flush=True)
        summaries[policy] = {
            population: {
                key: aggregate(values, key) for key in values[0]
            } for population, values in rows.items()
        }
        if policy_index == 0:
            reproduction = {
                'stored_labels_reproduced': True,
                'max_abs_cost_difference': maximum_label_difference,
                'rtol': 1e-4, 'atol': 1e-3,
            }
        del model
        torch.cuda.empty_cache()

    output = {
        'config': vars(args), 'memory_manifest': bank['manifest'],
        'num_scenes': len(stored), 'candidates_per_scene': count,
        'horizon': stored.shape[2], 'topk': topk,
        'old_model_reproduction': reproduction,
        'conditions': {
            'stored': 'Exact memory plans and old-model labels used by P1 training.',
            'fresh': 'Independent initial Gaussian plans, same states/goals/count/horizon.',
        },
        'limitation': 'Both conditions use training memory states and Gaussian plans; no CEM search or closed-loop success is measured.',
        'summary': summaries, 'per_scene': per_scene,
        'native_sources_file': str(source_out),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    print('[AUDIT_SUMMARY]', flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print('[NATIVE_PUSHT_CONFIG]', flush=True)
    for name, content in native['plan_configs'].items():
        if Path(name).stem == 'pusht':
            print('FILE:', name, '\n' + content, flush=True)
    print('[NATIVE_CEM_CONSTRUCTORS]', flush=True)
    print(json.dumps(native['cem_constructors'], ensure_ascii=False, indent=2), flush=True)
    print(f'[AUDIT_DONE] scenes={len(stored)}; models={len(policies)}; out={out}', flush=True)


if __name__ == '__main__':
    main()
