"""Explicitly install the current model/training code and confirmed history repair.

Without --apply, only report the proposed changes. Every original target is
backed up before the first write. Run against a stopped training checkout.
"""
import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess

if __package__:
    from .planning_history import policy_history_source, future_rollout_source
else:
    from planning_history import policy_history_source, future_rollout_source


def proposed_files(root):
    source = Path(__file__).resolve().parent
    policy = root / 'stable_worldmodel/policy.py'
    model = root / 'stable_worldmodel/wm/lewm/lewm.py'
    files = {
        root / 'stable_worldmodel/wm/lewm/module.py': (source / 'module.py').read_bytes(),
        root / 'scripts/train/lewm.py': (source / 'train_lewm.py').read_bytes(),
        policy: policy_history_source(policy.read_text(encoding='utf-8')).encode('utf-8'),
        model: future_rollout_source(model.read_text(encoding='utf-8')).encode('utf-8'),
    }
    for path, content in files.items():
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Expected an existing source file inside the SWM checkout: {path}')
        compile(content, str(path), 'exec')
    return files


def install(root, apply=False):
    root = Path(root).resolve()
    files = proposed_files(root)  # Validate both supported source layouts before writing anything.
    changes = {p: data for p, data in files.items() if p.read_bytes() != data}
    report = {'mode': 'apply' if apply else 'check', 'root': str(root),
              'changes': [str(p.relative_to(root)) for p in changes]}
    if not apply or not changes:
        return report
    backup = root / '.code_backups' / datetime.now().strftime('current_%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True, exist_ok=False)
    originals = {p: p.read_bytes() for p in changes}
    for path, data in originals.items():
        saved = backup / path.relative_to(root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        if saved.read_bytes() != data:
            raise RuntimeError(f'Backup verification failed: {saved}')
    versions = {}
    for name in ('stable-worldmodel', 'stable-pretraining', 'torch', 'torchvision',
                 'lightning', 'hydra-core', 'omegaconf', 'pylance', 'mujoco'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=False)
        report['upstream_commit'] = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        report['upstream_commit'] = None
    report.update(backup=str(backup), versions=versions)
    (backup / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        for path, content in changes.items():
            if path.read_bytes() != originals[path]:
                raise RuntimeError(f'Source changed after backup: {path}')
            path.write_bytes(content)
            if path.read_bytes() != content:
                raise RuntimeError(f'Installed copy verification failed: {path}')
    except Exception:
        for path, content in originals.items():
            path.write_bytes(content)
        raise
    report['status'] = 'installed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing stable-worldmodel checkout')
    parser.add_argument('--apply', action='store_true', help='Back up and apply the proposed changes')
    args = parser.parse_args()
    print(json.dumps(install(args.root, args.apply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
