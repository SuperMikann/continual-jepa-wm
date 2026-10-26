r"""Extract and plot training curves from stable-pretraining metrics.csv
or TensorBoard logs.

Usage:
    python plot_training_curves.py /path/to/metrics.csv
    python plot_training_curves.py /path/to/tensorboard/logdir

Examples:
    python plot_training_curves.py D:/Program/lewm/baseline_results/logs/metrics.csv
    python plot_training_curves.py D:/Program/lewm/baseline_results/logs/1e24a0455561

After running, it saves:
    - continual-wm/results/figures/baseline_loss_curves.png
    - continual-wm/results/figures/baseline_main_loss.png
    - continual-wm/results/figures/baseline_per_epoch_loss.png
    - continual-wm/results/data/baseline_scalars.csv
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def find_csv_or_logdir(user_input: str | None) -> Path:
    if user_input is not None:
        path = Path(user_input)
        if path.is_file() or path.is_dir():
            return path
        raise FileNotFoundError(f'Not found: {path}')

    root = Path(__file__).resolve().parent.parent / 'results'
    candidates = list(root.rglob('metrics.csv'))
    if not candidates:
        raise FileNotFoundError(
            'No metrics.csv found. Pass path to metrics.csv or logdir explicitly.'
        )
    return candidates[0]


def read_metrics_csv(csv_path: Path) -> dict[str, list]:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    columns = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                columns[key].append(float(value))
            except ValueError:
                columns[key].append(value)
    return columns


def read_tensorboard_logs(logdir: Path) -> dict[str, list]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as e:
        raise ImportError(
            'tensorboard is required. Install with: pip install tensorboard'
        ) from e

    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        raise ValueError(f'No scalar tags found in {logdir}')

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [e.value for e in events]
    data['step'] = [e.step for e in ea.Scalars(tags[0])]
    return data


def aggregate_per_epoch(data: dict[str, list], epoch_col: str) -> dict[str, list]:
    epochs = data[epoch_col]
    unique_epochs = sorted(set(int(e) for e in epochs if isinstance(e, (int, float))))

    aggregated = {'epoch': unique_epochs}
    for key in data:
        if key == epoch_col:
            continue

        per_epoch = []
        for ep in unique_epochs:
            values = [
                float(v)
                for e, v in zip(epochs, data[key])
                if e == ep and v not in (None, '') and not math.isnan(float(v))
            ]
            if values:
                per_epoch.append(mean(values))
            else:
                per_epoch.append(float('nan'))
        aggregated[key] = per_epoch

    return aggregated


def plot_curves(data: dict[str, list], output_dir: Path, per_epoch: dict[str, list]):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-step detailed curves
    x_col = 'epoch' if 'epoch' in data else 'step'
    x = data[x_col]

    metrics = {
        'fit/loss': 'Train Loss',
        'validate/loss_epoch': 'Validation Loss',
        'fit/pred_loss': 'Train Pred Loss',
        'validate/pred_loss_epoch': 'Validation Pred Loss',
        'fit/sigreg_loss': 'Train SigReg Loss',
        'validate/sigreg_loss_epoch': 'Validation SigReg Loss',
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics.items()):
        if key not in data:
            ax.set_title(f'{title} (not found)')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        y = [float(v) if v not in (None, '') else float('nan') for v in data[key]]
        ax.plot(x, y, linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Epoch' if x_col == 'epoch' else 'Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    x_numeric = [float(v) for v in x if v not in (None, '')]
    if x_numeric and len(x_numeric) > 20:
        max_x = int(max(x_numeric))
        for a in axes:
            a.set_xticks(range(0, max_x + 1, max(1, max_x // 6)))

    fig_path = output_dir / 'baseline_loss_curves.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f'Saved figure: {fig_path}')

    # Combined main loss curve (per-step)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if 'fit/loss' in data:
        y = [float(v) if v not in (None, '') else float('nan') for v in data['fit/loss']]
        ax.plot(x, y, label='Train Loss', linewidth=0.8, alpha=0.7)
    if 'validate/loss_epoch' in data:
        y = [float(v) if v not in (None, '') else float('nan') for v in data['validate/loss_epoch']]
        ax.plot(x, y, label='Validation Loss', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel('Epoch' if x_col == 'epoch' else 'Step')
    ax.set_ylabel('Loss')
    ax.set_title('LeWM Baseline Loss Curve (Push-T)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    x_numeric = [float(v) for v in x if v not in (None, '')]
    if x_numeric and len(x_numeric) > 20:
        max_x = int(max(x_numeric))
        ax.set_xticks(range(0, max_x + 1, max(1, max_x // 6)))

    main_path = output_dir / 'baseline_main_loss.png'
    fig.savefig(main_path, dpi=300)
    plt.close(fig)
    print(f'Saved figure: {main_path}')

    # Per-epoch clean curve (most useful for paper)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    epochs = per_epoch['epoch']
    train_key = 'fit/loss' if 'fit/loss' in per_epoch else None
    val_key = 'validate/loss_epoch' if 'validate/loss_epoch' in per_epoch else None

    if train_key:
        ax.plot(epochs, per_epoch[train_key], label='Train Loss', linewidth=1.5, marker='o', markersize=4)
    if val_key:
        ax.plot(epochs, per_epoch[val_key], label='Validation Loss', linewidth=1.5, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LeWM Baseline Per-Epoch Loss (Push-T)')
    ax.set_xticks(epochs[::2])
    ax.legend()
    ax.grid(True, alpha=0.3)
    per_epoch_path = output_dir / 'baseline_per_epoch_loss.png'
    fig.savefig(per_epoch_path, dpi=300)
    plt.close(fig)
    print(f'Saved figure: {per_epoch_path}')


def save_csv(data: dict[str, list], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'baseline_scalars.csv'

    keys = list(data.keys())
    n_rows = max(len(data[k]) for k in keys)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n_rows):
            row = []
            for key in keys:
                if i < len(data[key]):
                    row.append(data[key][i])
                else:
                    row.append('')
            writer.writerow(row)
    print(f'Saved CSV: {csv_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot LeWM baseline curves from metrics.csv or TensorBoard logs'
    )
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to metrics.csv or TensorBoard log directory',
    )
    args = parser.parse_args()

    input_path = find_csv_or_logdir(args.path)
    print(f'Reading from: {input_path}')

    if input_path.is_file() and input_path.suffix == '.csv':
        data = read_metrics_csv(input_path)
    elif input_path.is_dir():
        data = read_tensorboard_logs(input_path)
    else:
        raise ValueError(f'Unsupported input: {input_path}')

    print(f'Loaded {len(data[list(data.keys())[0]])} rows, columns: {list(data.keys())}')

    per_epoch = None
    if 'epoch' in data:
        per_epoch = aggregate_per_epoch(data, 'epoch')
        print(f'Aggregated {len(per_epoch["epoch"])} epochs')
    else:
        per_epoch = data

    project_root = Path(__file__).resolve().parent.parent
    figures_dir = project_root / 'results' / 'figures'
    data_dir = project_root / 'results' / 'data'

    plot_curves(data, figures_dir, per_epoch)
    save_csv(data, data_dir)


if __name__ == '__main__':
    main()
