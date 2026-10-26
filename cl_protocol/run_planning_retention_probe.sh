#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/stable-worldmodel
export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
mkdir -p logs results

teacher_policy=${1:-pusht_seed/weights_epoch_30.pt}
student_policy=${2:-lewm_pusht_t2_e30/weights_epoch_30.pt}

for policy in "$teacher_policy" "$student_policy"; do
  if [[ ! -f "$STABLEWM_HOME/checkpoints/$policy" ]]; then
    printf 'Missing checkpoint: %s\n' "$STABLEWM_HOME/checkpoints/$policy" >&2
    exit 2
  fi
done

python -u cl_protocol/planning_retention_probe.py --self-test

python -u cl_protocol/planning_retention_probe.py \
  --sequence A_pusht --task T1 \
  --teacher-policy "$teacher_policy" --student-policy "$teacher_policy" \
  --check --topk 2 --seed 0 \
  --out /root/autodl-tmp/stable-worldmodel/results/planning_retention_same_model_check.json

python - <<'PY'
import json
from pathlib import Path
path = Path('/root/autodl-tmp/stable-worldmodel/results/planning_retention_same_model_check.json')
data = json.loads(path.read_text())
assert len(data['per_scene']) == 1
row = data['per_scene'][0]
assert row['teacher_student_tau'] is not None, 'Teacher costs are flat; inspect the check.'
assert abs(row['teacher_student_tau'] - 1.0) < 1e-6, 'Same-model rankings differ.'
assert row['top1_agreement'] == 1.0, 'Same-model selected candidates differ.'
assert row['topk_overlap'] == 1.0
assert row['preference_kl'] < 1e-5, 'Same-model normalized costs differ.'
assert row['teacher_regret_std'] < 1e-5
print('[SAME_MODEL_CHECK_OK]', flush=True)
PY

python -u cl_protocol/planning_retention_probe.py \
  --sequence A_pusht --task T1 \
  --teacher-policy "$teacher_policy" --student-policy "$student_policy" \
  --num-scenes 20 --num-candidates 24 --topk 5 --seed 0 \
  --out /root/autodl-tmp/stable-worldmodel/results/planning_retention_pusht_t1_t2_seed0.json
