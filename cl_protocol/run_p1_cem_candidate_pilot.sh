#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/stable-worldmodel
export STABLEWM_HOME=/root/autodl-tmp/.stable_worldmodel
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
mkdir -p logs results

memory=results/p1_pusht_memory_cem_seed0.pt
run=p1_pusht_cem_score_kd_s0_300
check=p1_pusht_cem_score_kd_s0_check
teacher=pusht_seed/weights_epoch_30.pt
policy="$run/weights_step_300.pt"
printf '[CEM_PILOT_START] only candidate source changes; 300 steps\n'

if [[ ! -f "$memory" ]]; then
  python -u cl_protocol/prepare_cem_planning_memory.py
fi
python -u -c 'import torch; b=torch.load("results/p1_pusht_memory_cem_seed0.pt",map_location="cpu",weights_only=False); assert b["candidates"].shape[:3]==(64,8,5); assert b["manifest"]["candidate_source"]=="4 original Gaussian + 2 native CEM iteration 16 + 2 native CEM iteration 30"; assert b["manifest"]["gradient_check"]["gradient_norm"]["encoder"]>0; print("[MEMORY_CHECK_OK]",flush=True)'

if [[ ! -f "$STABLEWM_HOME/checkpoints/$check/pilot_result.json" ]]; then
  python -u cl_protocol/train_planning_retention.py \
    --mode score_kd --memory "$memory" --steps 5 --run-name "$check"
fi
python -u -c 'import json,os;from pathlib import Path;p=Path(os.environ["STABLEWM_HOME"])/"checkpoints/p1_pusht_cem_score_kd_s0_check";d=json.loads((p/"pilot_result.json").read_text());assert d["global_step"]==5 and len(d["trace"])==5;assert (p/"weights_step_5.pt").is_file();assert d["memory_manifest"]["candidate_source"]=="4 original Gaussian + 2 native CEM iteration 16 + 2 native CEM iteration 30";assert any(r["old_loss"]>0 for r in d["trace"]);print("[FIVE_STEP_CHECK_OK]",flush=True)'

if [[ ! -f "$STABLEWM_HOME/checkpoints/$run/pilot_result.json" ]]; then
  python -u cl_protocol/train_planning_retention.py \
    --mode score_kd --memory "$memory" --steps 300 --run-name "$run"
fi
python -u -c 'import json,os;from pathlib import Path;p=Path(os.environ["STABLEWM_HOME"])/"checkpoints/p1_pusht_cem_score_kd_s0_300";d=json.loads((p/"pilot_result.json").read_text());assert d["global_step"]==300 and len(d["trace"])==300;assert (p/"weights_step_300.pt").is_file();assert d["memory_manifest"]["candidate_source"]=="4 original Gaussian + 2 native CEM iteration 16 + 2 native CEM iteration 30";assert d["config"]["mode"]=="score_kd" and d["config"]["seed"]==0 and d["config"]["kd_weight"]==0.01 and d["config"]["temperature"]==1.0;print("[TRAIN_CHECK_OK]",flush=True)'

if [[ ! -f "results/${run}_retention.json" ]]; then
  python -u cl_protocol/planning_retention_probe.py \
    --sequence A_pusht --task T1 --teacher-policy "$teacher" \
    --student-policy "$policy" --seed 0 --num-scenes 20 --num-candidates 24 \
    --topk 5 --out "results/${run}_retention.json"
fi
if [[ ! -f "results/pred_error_A_pusht_${run}_weights_step_300.json" ]]; then
  (
    cd results
    python -u ../cl_protocol/pred_error.py --sequence A_pusht \
      --policy "$policy" --tasks T1 T2 --windows 200 --batch 32 --seed 0
  )
fi
if [[ ! -f results/p1_native_cem_retention_cem_memory.json ]]; then
  # Evaluate against the ORIGINAL memory and the same 4+4 diagnostic scenes.
  python -u cl_protocol/native_cem_retention_probe.py \
    --score-policy "$policy" --out results/p1_native_cem_retention_cem_memory.json \
    --other-policies p1_pusht_score_kd_s0_300/weights_step_300.pt \
      p1_pusht_finetune_s0_300/weights_step_300.pt \
      p1_pusht_replay_s0_300/weights_step_300.pt
fi

for task in T1 T2; do
  folder="/root/autodl-tmp/stable-worldmodel/results/p1_cem_behavior_${task}"
  mkdir -p "$folder"
  if [[ ! -f "$folder/forgetting_matrix_A_pusht.json" ]]; then
    (
      cd "$folder"
      python -u /root/autodl-tmp/stable-worldmodel/cl_protocol/eval_task.py \
        --eval-wm /root/autodl-tmp/stable-worldmodel/scripts/plan/eval_wm.py \
        --sequence A_pusht --matrix --tasks-sel "$task" \
        --num-eval 20 --seed 42 --policies "$policy"
    )
  fi
  python -u -c 'import json,sys;from pathlib import Path;task,policy,folder=sys.argv[1:];d=json.loads((Path(folder)/"forgetting_matrix_A_pusht.json").read_text());assert len(d)==1 and d[policy].get(task) is not None, "评估格失败";print("[TASK_CHECK_OK]",task,d,flush=True)' "$task" "$policy" "$folder"
done

python -u - <<'PY'
import json
import os
from pathlib import Path

root = Path('/root/autodl-tmp/stable-worldmodel')
results = root / 'results'
run = 'p1_pusht_cem_score_kd_s0_300'
policy = run + '/weights_step_300.pt'
training = json.loads((Path(os.environ['STABLEWM_HOME']) / 'checkpoints' / run / 'pilot_result.json').read_text())
probe = json.loads((results / f'{run}_retention.json').read_text())
prediction = json.loads((results / f'pred_error_A_pusht_{run}_weights_step_300.json').read_text())
cem = json.loads((results / 'p1_native_cem_retention_cem_memory.json').read_text())
assert training['global_step'] == 300 and len(probe['per_scene']) == 20
assert probe['config']['num_scenes'] == 20 and probe['config']['num_candidates'] == 24
assert cem['trace_check_passed'] and cem['initial_populations_equal']
assert cem['config']['score_policy'] == policy and len(cem['scene_metadata']) == 8
assert all(cem['guide_cost_reproduction_max_abs_diff'][p] is not None for p in (cem['config']['old_policy'], policy))
behavior = {}
for task in ('T1', 'T2'):
    matrix = json.loads((results / f'p1_cem_behavior_{task}' / 'forgetting_matrix_A_pusht.json').read_text())
    assert len(matrix) == 1 and matrix[policy].get(task) is not None, f'{task}: failed behavior cell'
    behavior[task] = matrix[policy][task]
old_rows = json.loads((results / 'p1_pusht_pilot_summary.json').read_text())['rows']
previous_behavior = {}
for task in ('T1', 'T2'):
    previous_behavior[task] = json.loads((results / f'p1_behavior_{task}' / 'forgetting_matrix_A_pusht.json').read_text())
summary = {
    'candidate_source': training['memory_manifest']['candidate_source'],
    'steps': training['global_step'],
    'training_seconds': training['training_seconds'],
    'peak_gpu_bytes': training['peak_gpu_bytes'],
    'memory_tensor_bytes': training['memory_manifest']['tensor_bytes'],
    'memory_preparation_seconds': training['memory_manifest']['preparation_seconds'],
    'retention': probe['summary'],
    'prediction': prediction['results'],
    'returned_plan_summary': cem['returned_plan_summary'],
    'behavior': behavior,
    'previous_offline_controls': old_rows,
    'previous_behavior': previous_behavior,
    'status': 'single-seed 300-step candidate-source pilot; 20 behavior episodes per task',
}
path = results / 'p1_pusht_cem_pilot_summary.json'
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
print(f'[CEM_PILOT_DONE] summary={path}', flush=True)
PY
