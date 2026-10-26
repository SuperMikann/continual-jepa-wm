#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/stable-worldmodel
export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
mkdir -p logs results

stage=${1:-check}
case "$stage" in check|train|all) ;; *) printf 'Use check, train or all\n' >&2; exit 2 ;; esac
printf '[P1_START] stage=%s\n' "$stage"
memory=results/p1_pusht_memory_seed0.pt
teacher=pusht_seed/weights_epoch_30.pt
test -f "$STABLEWM_HOME/checkpoints/$teacher"
if [[ ! -f scripts/train/config/lewm.yaml ]]; then
  printf 'Missing scripts/train/config/lewm.yaml; pass the real native config directory to the training entry.\n' >&2
  exit 2
fi

if [[ ! -f "$memory" ]]; then
  python -u cl_protocol/train_planning_retention.py --mode prepare --teacher "$teacher" --memory "$memory"
fi

if [[ "$stage" == check || "$stage" == all ]]; then
  for mode in finetune replay score_kd; do
    python -u cl_protocol/train_planning_retention.py \
      --mode "$mode" --teacher "$teacher" --memory "$memory" \
      --steps 5 --run-name "p1_pusht_${mode}_s0_check"
  done
  python -u - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['STABLEWM_HOME']) / 'checkpoints'
for mode in ('finetune', 'replay', 'score_kd'):
    folder = root / f'p1_pusht_{mode}_s0_check'
    d = json.loads((folder / 'pilot_result.json').read_text())
    assert d['global_step'] == 5
    assert (folder / 'weights_step_5.pt').is_file()
    assert len(d['trace']) == 5
    assert d['memory_manifest']['gradient_check']['gradient_norm']['encoder'] > 0
    if mode != 'finetune':
        assert any(row['old_loss'] > 0 for row in d['trace']), f'{mode}: old loss not exercised'
print('[THREE_ARM_CHECK_OK]', flush=True)
PY
fi

if [[ "$stage" == train || "$stage" == all ]]; then
  # Require actual five-step results; a wrapper echo is not a completion gate.
  for mode in finetune replay score_kd; do
    test -f "$STABLEWM_HOME/checkpoints/p1_pusht_${mode}_s0_check/pilot_result.json"
  done
  (
    cd results
    python -u ../cl_protocol/pred_error.py --sequence A_pusht \
      --policy "$teacher" --tasks T1 T2 --windows 200 --batch 32 --seed 0
  )
  for mode in finetune replay score_kd; do
    run="p1_pusht_${mode}_s0_300"
    python -u cl_protocol/train_planning_retention.py \
      --mode "$mode" --teacher "$teacher" --memory "$memory" \
      --steps 300 --run-name "$run"
    python -u cl_protocol/planning_retention_probe.py \
      --sequence A_pusht --task T1 --teacher-policy "$teacher" \
      --student-policy "$run/weights_step_300.pt" --seed 0 \
      --num-scenes 20 --num-candidates 24 --topk 5 \
      --out "results/${run}_retention.json"
    (
      cd results
      python -u ../cl_protocol/pred_error.py --sequence A_pusht \
        --policy "$run/weights_step_300.pt" --tasks T1 T2 \
        --windows 200 --batch 32 --seed 0
    )
  done
  python -u - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['STABLEWM_HOME']) / 'checkpoints'
results = Path('/root/autodl-tmp/stable-worldmodel/results')
rows = []
for mode in ('finetune', 'replay', 'score_kd'):
    run = f'p1_pusht_{mode}_s0_300'
    training = json.loads((root / run / 'pilot_result.json').read_text())
    probe = json.loads((results / f'{run}_retention.json').read_text())
    prediction = json.loads((results / f'pred_error_A_pusht_{run}_weights_step_300.json').read_text())
    assert training['global_step'] == 300
    assert Path(training['checkpoint']).is_file()
    assert len(probe['per_scene']) == 20
    row = {'mode': mode, 'steps': training['global_step'], 'training_seconds': training['training_seconds'], 'peak_gpu_bytes': training['peak_gpu_bytes']}
    row.update({key: probe['summary'][key]['mean'] for key in ('teacher_student_tau', 'top1_agreement', 'topk_overlap', 'teacher_regret_std', 'preference_kl')})
    row.update({f'{task}_pred_mse': prediction['results'][task]['mse_mean'] for task in ('T1', 'T2')})
    rows.append(row)
baseline = json.loads((results / 'pred_error_A_pusht_pusht_seed_weights_epoch_30.json').read_text())
output = {'teacher_prediction': baseline['results'], 'rows': rows, 'status': 'offline 300-step screening only; closed-loop old/new behavior pending'}
path = results / 'p1_pusht_pilot_summary.json'
path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
print(f'[PILOT_DONE] summary={path}', flush=True)
PY
fi
