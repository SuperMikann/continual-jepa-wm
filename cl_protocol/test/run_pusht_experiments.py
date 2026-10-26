"""PushT controlled training and diagnostics; preserves the historical CLI modes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

if __package__:
    from .pusht.training import (prepare_protocol, prepare_configs, evaluate_arm, paired_comparison,
                                 run_t2_baseline, run_t2_v2, run_t2_structure, run_t2_bn_delta)
    from .pusht.diagnostics import run_t2_bn_probe, run_t2_planning_probe, run_t2_history
    from .pusht.adapters import run_adapter_t2
else:
    from pusht.training import (prepare_protocol, prepare_configs, evaluate_arm, paired_comparison,
                                run_t2_baseline, run_t2_v2, run_t2_structure, run_t2_bn_delta)
    from pusht.diagnostics import run_t2_bn_probe, run_t2_planning_probe, run_t2_history
    from pusht.adapters import run_adapter_t2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--tag', default='pusht_v2_smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--mode', choices=['smoke', 'compare', 't2-baseline', 't2-v2', 't2-schedule', 't2-structure', 't2-bn-delta', 't2-bn-probe', 't2-planning-probe', 't2-history', 'adapter-t2'], default='smoke')
    p.add_argument('--base-policy', default=None)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--baseline-tag', default='pusht_t2_baseline_20261023')
    p.add_argument('--control-tag', default=None)
    p.add_argument('--comparison-tag', default='pusht_t2_bn_delta_20261024')
    p.add_argument('--reuse-first-cell-from', default=None)
    p.add_argument('--windows', type=int, default=128)
    p.add_argument('--planning-scenes', type=int, default=4)
    p.add_argument('--candidate-samples', type=int, default=12)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--num-eval', type=int, default=100)
    args = p.parse_args()
    if min(args.steps, args.batch, args.num_eval) < 1:
        p.error('steps/batch/num-eval must be positive')
    if Path(args.tag).name != args.tag or not args.tag or args.tag in ('.', '..'):
        p.error('tag must be a directory name')
    root = args.root.resolve()
    home = Path(os.environ.get('STABLEWM_HOME', '/root/autodl-tmp/.stable_worldmodel'))
    if args.mode == 'adapter-t2':
        if not args.base_policy or args.epochs < 1:
            p.error('adapter-t2 requires an actual --base-policy and positive --epochs')
        run_adapter_t2(root, home, args.tag, args.base_policy, args.epochs, args.num_eval)
        return
    if args.mode == 't2-history':
        if Path(args.comparison_tag).name != args.comparison_tag or args.comparison_tag in ('', '.', '..'):
            p.error('comparison-tag must be a directory name')
        if args.reuse_first_cell_from is not None and (Path(args.reuse_first_cell_from).name != args.reuse_first_cell_from
                or args.reuse_first_cell_from in ('', '.', '..', args.tag)):
            p.error('reuse-first-cell-from must name a different results directory')
        run_t2_history(root, home, args.tag, args.comparison_tag, args.num_eval,
                       reuse_first_cell_from=args.reuse_first_cell_from)
        return
    if args.mode == 't2-planning-probe':
        if (Path(args.comparison_tag).name != args.comparison_tag or args.comparison_tag in ('', '.', '..')
                or min(args.windows, args.planning_scenes) < 1 or args.candidate_samples < 3
                or args.planning_scenes > args.windows):
            p.error('Use a comparison directory name, positive counts, scenes <= windows and >= 3 candidates')
        run_t2_planning_probe(root, home, args.tag, args.comparison_tag, args.windows,
                              args.batch, args.planning_scenes, args.candidate_samples)
        return
    if args.mode == 't2-bn-probe':
        if (Path(args.comparison_tag).name != args.comparison_tag
                or args.comparison_tag in ('', '.', '..') or args.windows < 2):
            p.error('comparison-tag must be a directory name; windows must be >= 2')
        run_t2_bn_probe(root, home, args.tag, args.comparison_tag, args.windows, args.batch)
        return
    if args.mode in ('t2-structure', 't2-bn-delta'):
        control_tag = args.control_tag or ('pusht_t2_structure_20261024_no_pred_proj'
                                         if args.mode == 't2-bn-delta' else 'pusht_t2_schedule_20261024_step')
        if Path(control_tag).name != control_tag or control_tag in ('', '.', '..'):
            p.error('control-tag must be a directory name')
        runner = run_t2_bn_delta if args.mode == 't2-bn-delta' else run_t2_structure
        runner(root, home, args.tag, args.num_eval, control_tag)
        return
    if args.mode == 't2-baseline':
        run_t2_baseline(root, home, args.tag, args.num_eval)
        return
    if args.mode in ('t2-v2', 't2-schedule'):
        if Path(args.baseline_tag).name != args.baseline_tag:
            p.error('baseline-tag must be a directory name')
        run_t2_v2(root, home, args.tag, args.num_eval, args.baseline_tag,
                  scheduler_only=args.mode == 't2-schedule')
        return
    data = home / 'datasets/pusht_expert_train.lance'
    if not (data / '_versions').is_dir():
        raise FileNotFoundError(f'Local PushT dataset missing: {data}; no download attempted')
    output = root / 'results' / args.tag
    output.mkdir(parents=True, exist_ok=False)
    comparison = args.mode == 'compare'
    summary = dict(status='preparing', scope='Fixed-budget initial comparison; not proof of convergence or continual retention'
                   if comparison else '20 updates per arm; plumbing only, not component efficacy',
                   dataset=str(data), seed=3072, steps=args.steps if comparison else 20,
                   batch=args.batch if comparison else 16, num_eval=args.num_eval if comparison else 0, rows=[])
    def save():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save()
    try:
        env = dict(os.environ, STABLEWM_HOME=str(home), PYTHONUNBUFFERED='1')
        env.pop('LOCAL_DATASET_DIR', None)
        env.update(MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa')
        os.environ['STABLEWM_HOME'] = str(home)
        os.environ.pop('LOCAL_DATASET_DIR', None)
        stats = prepare_protocol(home, output, args.num_eval) if comparison else None
        configs = prepare_configs(root, output, args.tag, mode=args.mode, steps=args.steps, batch=args.batch, stats=stats)
        for arm, run, config in configs:
            summary['status'] = f'running_{arm}'
            save()
            label = 'COMPARE_ARM_START' if comparison else 'SMOKE_ARM_START'
            print(f'[{label}] {arm} run={run} steps={summary["steps"]}', flush=True)
            command = [sys.executable, '-u', str(root / 'scripts/train/lewm.py'),
                       '--config-path', str(output), '--config-name', config.stem]
            subprocess.run(command, cwd=root, env=env, check=True)
            receipt = home / 'checkpoints' / run / ('comparison_train.json' if comparison else 'real_smoke_check.json')
            row = json.loads(receipt.read_text())
            assert row['status'] == 'passed' and row['global_step'] == summary['steps']
            summary['rows'].append(dict(arm=arm, run=run, receipt=str(receipt), **row))
            save()
            if comparison:
                summary['status'] = f'evaluating_{arm}'
                save()
                summary['rows'][-1]['behavior'] = evaluate_arm(root, output, home, env, arm, run, args.num_eval)
                save()
            print(f'[{"COMPARE_ARM_OK" if comparison else "SMOKE_ARM_OK"}] {arm}', flush=True)
        if comparison:
            summary['comparison'] = paired_comparison(summary['rows'])
        summary['status'] = 'complete'
        save()
        print('[PUSHT_V2_COMPARE_DONE]' if comparison else '[PUSHT_V2_SMOKE_DONE]', output / 'summary.json', flush=True)
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
