"""Reacher dynamics screening, adaptation and sequential comparison."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

if __package__:
    from .reacher.common import CASES
    from .reacher.pipeline import batch, adaptation_batch, comparison_batch, stage3_batch
    from .reacher.training import collection_worker, training_worker
    from .reacher.evaluation import worker, prediction_worker
else:
    from reacher.common import CASES
    from reacher.pipeline import batch, adaptation_batch, comparison_batch, stage3_batch
    from reacher.training import collection_worker, training_worker
    from reacher.evaluation import worker, prediction_worker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/stable-worldmodel'))
    ap.add_argument('--mode', choices=('screen', 'adapt', 'compare', 'stage3', 'collect', 'train', 'prediction'), default='screen')
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--policy', default='m3g_reacher_t1/weights_epoch_30.pt')
    ap.add_argument('--dataset', default='dmc/reacher_random.lance')
    ap.add_argument('--num-eval', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--scene-manifest', type=Path)
    ap.add_argument('--run-id')
    ap.add_argument('--base-run', type=Path)
    ap.add_argument('--comparison-run', type=Path)
    ap.add_argument('--methods', choices=('replay', 'growth'), nargs='+', default=['replay', 'growth'])
    ap.add_argument('--train-episodes', type=int, default=500)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--pred-episodes', type=int, default=50)
    ap.add_argument('--pred-windows', type=int, default=512)
    ap.add_argument('--pred-batch', type=int, default=32)
    ap.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--cases', choices=tuple(CASES), nargs='+', default=list(CASES))
    ap.add_argument('--case', choices=tuple(CASES), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.num_eval is None:
        args.num_eval = 100 if args.mode in ('adapt', 'compare', 'stage3') else 20
    if args.seed is None:
        args.seed = 20261023 if args.mode in ('adapt', 'compare', 'stage3') else 20261022
    if args.num_eval <= 0:
        ap.error('--num-eval must be positive')
    if args.epochs <= 0 or args.train_episodes < 10 or args.train_episodes % 10:
        ap.error('--epochs must be positive; --train-episodes must be a positive multiple of 10')
    if args.pred_episodes < 10 or args.pred_episodes % 10 or args.pred_windows <= 0 or args.pred_batch <= 0:
        ap.error('Prediction episodes must be a positive multiple of 10; windows/batch must be positive')
    if args.mode in ('collect', 'train', 'prediction') and args.config is None:
        ap.error('--config is required for the internal collection/training worker')
    args.root = args.root.resolve()
    if args.run_id is None:
        args.run_id = {'compare': 'reacher_damping2_compare_20261022',
                       'stage3': 'reacher_damping_t3_20261022'}.get(args.mode, 'reacher_damping2_adapt_20261022')
    args.base_run = (args.base_run or args.root/'results/reacher_damping2_adapt_20261022').resolve()
    args.comparison_run = (args.comparison_run or args.root/'results/reacher_damping2_compare_20261022').resolve()
    args.methods = list(dict.fromkeys(args.methods))
    if args.scene_manifest is not None:
        args.scene_manifest = args.scene_manifest.resolve()
        if not args.scene_manifest.is_file():
            ap.error('Scene manifest does not exist')
    if args.config is not None:
        args.config = args.config.resolve()
    default_name = args.run_id if args.mode in ('adapt', 'compare', 'stage3') else 'reacher_dynamics_screen_20261022'
    args.out_dir = (args.out_dir or args.root / 'results'/default_name).resolve()
    args.cases = list(dict.fromkeys(['baseline', *args.cases]))
    if args.mode == 'adapt':
        adaptation_batch(args)
    elif args.mode == 'compare':
        comparison_batch(args)
    elif args.mode == 'stage3':
        stage3_batch(args)
    elif args.mode == 'collect':
        collection_worker(args)
    elif args.mode == 'train':
        training_worker(args)
    elif args.mode == 'prediction':
        prediction_worker(args)
    elif args.case:
        worker(args)
    else:
        batch(args)


if __name__ == '__main__':
    main()
