"""任务边界处的离线误差比值检测；只报告判定，不自动增长或训练。

修复版要求 --action-stats 提供当前模型固定输入变换的 mean/scale。
旧的逐任务拟合方式仅在 --legacy 下启用。最后 N 条轨迹不是天然留出集；
可用 --split-manifest 提供实际训练/检测 episode 清单并检查二者不重叠。
历史阈值 1.15 保留作诊断参数，未证明其无漂移误报率或跨任务稳健性。
--calibrate 用分开的参考/校准/验证轨迹池定阈值；阈值冻结后才评估验证池。
文件格式和运行说明见本目录 README.md。
"""

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from pred_error import FixedActionScaler, episode_col, task_pred_error


def drift_decision(err_ref, err_probe, tau):
    if not all(math.isfinite(x) for x in (err_ref, err_probe, tau)):
        raise ValueError('误差与阈值必须为有限数值')
    if err_ref <= 0 or err_probe < 0 or tau <= 0:
        raise ValueError('参考误差和阈值必须 > 0，探测误差必须 >= 0')
    score = err_probe / err_ref
    if not math.isfinite(score):
        raise ValueError('误差比值溢出，拒绝判定')
    return score, score > tau


def load_action_stats(path, policy):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('policy') != policy or not data.get('source'):
        raise ValueError('动作统计必须注明匹配的 policy 和真实来源 source')
    scaler = FixedActionScaler(data['mean'], data['scale'])
    record = {
        'mode': 'fixed', 'policy': policy, 'source': data['source'],
        'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist(),
    }
    return scaler, record


def episode_ids(values, name, allow_empty=False):
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(type(v) is not int for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'{name} 必须为无重复的整数 episode 列表')
    return values


def select_episodes(dataset, dataset_name, role, reserve_last, manifest=None):
    """检查所给清单；无法凭空验证用户提供的训练清单是否完整。"""
    if reserve_last <= 0:
        raise ValueError('reserve-last 必须 > 0')
    ep_arr = np.asarray(dataset.get_col_data(episode_col(dataset)))
    available = np.unique(ep_arr)
    metadata = {'dataset': dataset_name, 'independence': 'unverified'}
    if manifest is None:
        selected = available[-reserve_last:]
        metadata['selection'] = 'last_n_episodes_not_guaranteed_heldout'
    else:
        item = manifest[role]
        if item.get('dataset') != dataset_name or not item.get('source'):
            raise ValueError(f'{role} 清单的数据集不匹配或缺少 source')
        selected = episode_ids(item['evaluation_episodes'], f'{role} evaluation')
        trained = episode_ids(item['training_episodes'], f'{role} training', allow_empty=True)
        if set(selected) & set(trained):
            raise ValueError(f'{role} 检测 episode 与训练清单重叠')
        if not set(selected).issubset(set(available.tolist())):
            raise ValueError(f'{role} 检测 episode 不在数据集中')
        metadata.update({
            'selection': 'explicit_manifest',
            'independence': 'disjoint_from_supplied_training_list',
            'manifest_source': item['source'],
            'training_episode_count': len(trained),
            'training_manifest_completeness_verified': False,
        })
    if not len(selected):
        raise ValueError(f'{role} 没有可检测轨迹')
    selected = np.asarray(selected, dtype=available.dtype)
    steps = np.asarray(dataset.get_col_data('step_idx'))
    lengths = {e: int(steps[ep_arr == e].max()) + 1 for e in selected}
    metadata['episode_ids'] = selected.tolist()
    return selected, lengths, metadata


def task_error(model, dataset, dataset_name, role, args, device, scaler, manifest):
    episodes, lengths, metadata = select_episodes(
        dataset, dataset_name, role, args.reserve_last, manifest
    )
    errors = task_pred_error(
        model, dataset, episodes, lengths, args, device, action_scaler=scaler
    )
    return errors, metadata


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks-cfg', default=str(Path(__file__).parent / 'tasks.yaml'))
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--ref-task', required=True)
    ap.add_argument('--probe-task', required=True)
    ap.add_argument('--tau', type=float, default=1.15,
                    help='历史诊断阈值；尚未验证误报率，默认 1.15')
    ap.add_argument('--reserve-last', type=int, default=200)
    ap.add_argument('--windows', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--screen-repeats', type=int, default=0,
                      help='小范围真实数据检查次数；0保留原单次入口，不训练或自动扩容')
    mode.add_argument('--calibrate', action='store_true', help='分开轨迹池校准阈值，再用未参与校准的轨迹验证')
    ap.add_argument('--calibration-batches', type=int, default=40)
    ap.add_argument('--validation-batches', type=int, default=20)
    ap.add_argument('--reference-windows', type=int, default=512)
    ap.add_argument('--pool-episodes', type=int, default=200, help='校准模式每个独立轨迹池的episode数')
    ap.add_argument('--target-fpr', type=float, default=.05, help='预设校准上尾概率；不是已验证的真实误报率')
    ap.add_argument('--exclude-screen-record', type=Path, help='校准时排除先前screen用过的整个episode池')
    ap.add_argument('--previous-calibration', type=Path,
                    help='排除上轮全部T1轨迹；按上轮池种子复用T2验证池并逐ID核对，不用于定阈值')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--action-stats', help='模型固定动作 mean/scale 的 JSON')
    source.add_argument('--legacy', action='store_true', help='仅复现历史逐任务归一化')
    ap.add_argument('--split-manifest', help='实际训练/检测 episode 清单 JSON')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    if min(args.reserve_last, args.windows, args.batch, args.stride) <= 0:
        ap.error('reserve-last/windows/batch/stride 必须 > 0')
    if not math.isfinite(args.tau) or args.tau <= 0:
        ap.error('tau 必须有限且 > 0')
    if args.seed < 0 or args.screen_repeats < 0:
        ap.error('seed/screen-repeats 必须 >= 0')
    if args.screen_repeats and (args.legacy or args.ref_task == args.probe_task):
        ap.error('screen需要固定动作统计和两个不同条件的任务名')
    if args.previous_calibration and not args.calibrate:
        ap.error('previous-calibration仅用于calibrate模式')
    if args.calibrate:
        if args.legacy or args.split_manifest or args.ref_task == args.probe_task:
            ap.error('calibrate需要固定动作统计及不同任务；本模式自行分开轨迹池，不使用旧split-manifest')
        if min(args.calibration_batches, args.validation_batches, args.reference_windows, args.pool_episodes) < 1:
            ap.error('校准/验证批数、参考窗口及轨迹池大小必须为正')
        if not math.isfinite(args.target_fpr) or not 0 < args.target_fpr < 1:
            ap.error('target-fpr必须在0和1之间')
        if math.ceil((args.calibration_batches+1)*(1-args.target_fpr)) > args.calibration_batches:
            ap.error('校准批数不足以估计指定上分位点，请增加批数或调整预设target-fpr')
    return args


def save_record(out, record):
    history = json.loads(out.read_text(encoding='utf-8')) if out.exists() else []
    if not isinstance(history, list):
        history = [history]
    if any(r.get('protocol') != record['protocol'] for r in history):
        raise ValueError('拒绝向不同检测协议的历史文件追加；请使用新输出路径')
    history.append(record)
    serialized = json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(serialized, encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    scaler, action_metadata = (None, {'mode': 'legacy_per_task_fit'})
    if args.action_stats:
        scaler, action_metadata = load_action_stats(args.action_stats, args.policy)
    manifest = None
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding='utf-8'))
        if manifest.get('policy') != args.policy:
            raise ValueError('训练/检测清单必须对应当前 policy 及其全部训练历史')

    with open(args.tasks_cfg, encoding='utf-8') as f:
        seq = yaml.safe_load(f)['sequences'][args.sequence]
    all_tasks = {**seq['tasks'], **seq.get('probe_tasks', {})}
    for name in (args.ref_task, args.probe_task):
        if name not in all_tasks:
            raise KeyError(f'未知任务 {name}，可用: {sorted(all_tasks)}')

    import stable_worldmodel as swm

    if args.calibrate:
        return run_calibration(args, swm, all_tasks, scaler, action_metadata)
    if args.screen_repeats:
        return run_screen(args, swm, all_tasks, scaler, action_metadata, manifest)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    print(f'[模型] {args.policy} 已加载（{device}）')
    print('[检测说明] 仅输出离线判定，不执行增长或训练；阈值稳健性待验证。')
    if manifest is None:
        print('[数据说明] 最后 N 条轨迹与训练的独立性未核实，结果仅作诊断。')
    else:
        print('[数据说明] 检查所给清单不重叠；清单完整性需以训练记录核实。')
    if args.legacy:
        print('[兼容模式] 两组各自拟合动作统计，仅供复现旧结果。')

    results, selections = {}, {}
    for role, task in (('ref', args.ref_task), ('probe', args.probe_task)):
        name = all_tasks[task]['dataset']
        dataset = swm.data.load_dataset(name, keys_to_cache=['pixels', 'action'])
        results[role], selections[role] = task_error(
            model, dataset, name, role, args, device, scaler, manifest
        )
    err_ref, std_ref, n_ref = results['ref']
    err_probe, std_probe, n_probe = results['probe']
    score, triggered = drift_decision(err_ref, err_probe, args.tau)
    protocol = 'legacy_per_task_v1' if args.legacy else 'fixed_action_v2'
    record = {
        'protocol': protocol, 'policy': args.policy, 'sequence': args.sequence,
        'ref_task': args.ref_task, 'probe_task': args.probe_task,
        'err_ref': err_ref, 'err_probe': err_probe, 's': score,
        'std_ref': std_ref, 'std_probe': std_probe, 'n_ref': n_ref, 'n_probe': n_probe,
        'tau': args.tau, 'triggered': triggered, 'decision_only': True,
        'threshold_validated': False,
        'reserve_last': args.reserve_last, 'windows': args.windows,
        'batch': args.batch, 'stride': args.stride, 'seed': args.seed,
        'action_normalization': action_metadata, 'data_selection': selections,
    }
    safe = args.policy.replace('/', '_').replace(chr(92), '_').replace('.pt', '')
    out = (args.out or Path(f'drift_trigger_{protocol}_{args.sequence}_{safe}.json')).resolve()
    save_record(out, record)
    print(f'[drift] {args.ref_task}: {err_ref:.6g} ± {std_ref:.6g} (n={n_ref})')
    print(f'[drift] {args.probe_task}: {err_probe:.6g} ± {std_probe:.6g} (n={n_probe})')
    print('DRIFT_RESULT ' + json.dumps(record, ensure_ascii=False, allow_nan=False))
    print(f'[判定] {"超过" if triggered else "未超过"}阈值；仅记录，不自动增长')
    print(f'[完成] 已保存: {out}')


def calibrated_threshold(scores, target_fpr):
    """Pre-specified upper order statistic; no shifted/validation scores enter."""
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or (values < 0).any() or not math.isfinite(target_fpr) or not 0 < target_fpr < 1):
        raise ValueError('校准分数/目标上尾概率无效')
    rank = math.ceil((len(values)+1)*(1-target_fpr))
    if rank > len(values):
        raise ValueError('校准分数数量不足')
    return dict(tau=max(1., float(np.sort(values)[rank-1])), rank=rank,
                count=len(values), target_fpr=target_fpr,
                rule='max(1, sorted_null_scores[ceil((n+1)*(1-alpha))-1])',
                scope='Empirical calibration, not a proven cross-condition false-alarm guarantee')


def calibration_pools(dataset, count, span, seed, excluded=()):
    """Disjoint episode pools; raw episode ids and native load_chunk ordinals are recorded."""
    ep = np.asarray(dataset.get_col_data(episode_col(dataset)))
    steps = np.asarray(dataset.get_col_data('step_idx'))
    ids, inverse = np.unique(ep, return_inverse=True)
    maximum = np.full(len(ids), -1, dtype=np.int64)
    np.maximum.at(maximum, inverse, steps)
    lengths = maximum+1
    eligible = np.flatnonzero((lengths > span) & ~np.isin(ids, list(excluded)))
    needed = sum(count.values())
    if len(eligible) < needed:
        raise ValueError(f'新轨迹池不足：需要{needed}，可用{len(eligible)}；不自动复用旧池')
    picked = np.random.default_rng(seed).choice(eligible, needed, replace=False)
    result, offset = {}, 0
    for name, n in count.items():
        ordinals = np.sort(picked[offset:offset+n])
        offset += n
        result[name] = dict(episode_ordinals=ordinals.tolist(), episode_ids=ids[ordinals].tolist())
    return result, {int(i):int(length) for i,length in enumerate(lengths)}


def run_calibration(args, swm, all_tasks, scaler, action_metadata):
    safe = args.policy.replace('/', '_').replace(chr(92), '_').replace('.pt', '')
    out = (args.out or Path(f'drift_calibration_{args.sequence}_{safe}.json')).resolve()
    if out.exists():
        raise FileExistsError(f'保留已有结果，请用新路径：{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(status='preparing', protocol='fixed_action_calibrated_v1', policy=args.policy,
        sequence=args.sequence, ref_task=args.ref_task, probe_task=args.probe_task,
        action_normalization=action_metadata, seed=args.seed, windows=args.windows,
        batch=args.batch, stride=args.stride, reference_windows=args.reference_windows,
        pool_episodes=args.pool_episodes, calibration_batches=args.calibration_batches,
        validation_batches=args.validation_batches, target_fpr=args.target_fpr,
        legacy_tau=args.tau, threshold_calibrated=False, threshold_validated=False,
        training_independence='unverified', decision_only=True, calibration_rows=[], validation_rows=[],
        scope='Disjoint reference/calibration/validation episode pools; no model training or growth. '
              'Repeated window batches reuse their own pool. Training independence is unverified; '
              'the target tail probability is a design setting, not a validated guarantee. '
              'Changed data can differ in scene/action/source as well as appearance.')
    def save():
        result['elapsed_seconds'] = time.monotonic()-started
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save()
    print('DRIFT_CAL_START', flush=True)
    try:
        names = {role:all_tasks[task]['dataset'] for role,task in
                 (('ref',args.ref_task),('probe',args.probe_task))}
        if names['ref'] == names['probe']:
            raise ValueError('校准验证需明确不同的数据集条件')
        excluded = {'ref':[], 'probe':[]}
        if args.exclude_screen_record:
            prior = json.loads(args.exclude_screen_record.read_text(encoding='utf-8'))
            if (prior.get('status') != 'complete' or prior.get('protocol') != 'fixed_action_v2_screen'
                    or prior.get('policy') != args.policy):
                raise ValueError('待排除screen记录的完成状态/协议/权重不匹配')
            for role in excluded:
                entry = prior['data_selection'][role]
                if entry['dataset'] != names[role]:
                    raise ValueError('待排除screen数据集不匹配')
                excluded[role] = entry['episode_ids']+entry.get('excluded_short_episode_ids',[])
            result['excluded_screen_record'] = str(args.exclude_screen_record.resolve())
        else:
            result['excluded_screen_record'] = None
        previous = None
        result['previous_calibration_record'] = None
        result['normal_pools_disjoint_from_previous'] = False
        result['changed_validation_reused'] = False
        if args.previous_calibration:
            previous = json.loads(args.previous_calibration.read_text(encoding='utf-8'))
            expected = dict(status='complete', protocol='fixed_action_calibrated_v1', policy=args.policy,
                            sequence=args.sequence, ref_task=args.ref_task, probe_task=args.probe_task,
                            stride=args.stride, pool_episodes=args.pool_episodes,
                            action_normalization=action_metadata)
            if any(previous.get(key) != value for key,value in expected.items()):
                raise ValueError('上轮校准记录的状态/权重/任务/步长/动作统计/池大小不匹配')
            for name in ('reference','calibration','validation','changed_validation'):
                pool = previous['episode_pools'][name]
                role = 'probe' if name=='changed_validation' else 'ref'
                if pool['dataset'] != names[role]:
                    raise ValueError('上轮轨迹池的数据集不匹配')
                ids = episode_ids(pool['episode_ids'],name)
                if len(ids) != args.pool_episodes:
                    raise ValueError('上轮轨迹池数量不匹配')
                if role=='ref':
                    excluded['ref'].extend(ids)
            result['previous_calibration_record'] = str(args.previous_calibration.resolve())
            result['scope'] += (' Previous normal episode pools are excluded. The changed-condition '
                                'pool is reused explicitly; this is not a fresh changed-condition test. '
                                'Normal pools differ between runs, so this is not a paired window-size comparison.')
        datasets = {}
        for role,name in names.items():
            local = swm.data.utils.get_cache_dir(sub_folder='datasets')/name
            if not local.exists():
                raise FileNotFoundError(f'本地数据缺失，不下载：{local}')
            datasets[role] = swm.data.load_dataset(str(local), frameskip=1, num_steps=1,
                keys_to_load=['pixels','action'], keys_to_cache=['action'])
        checkpoint = swm.data.utils.get_cache_dir(sub_folder='checkpoints')/args.policy
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = swm.wm.utils.load_pretrained(str(checkpoint)).to(device).eval()
        model.requires_grad_(False)
        span = int(getattr(model.predictor,'num_frames',3))*args.stride
        t1_pools, t1_lengths = calibration_pools(datasets['ref'],
            dict(reference=args.pool_episodes, calibration=args.pool_episodes, validation=args.pool_episodes),
            span, args.seed, excluded['ref'])
        changed_pool_seed = (int(previous['seed']) if previous is not None else args.seed)+1
        t2_pools, t2_lengths = calibration_pools(datasets['probe'],
            dict(changed_validation=args.pool_episodes),span,changed_pool_seed,excluded['probe'])
        if previous is not None:
            if (t2_pools['changed_validation']['episode_ids'] !=
                    previous['episode_pools']['changed_validation']['episode_ids']):
                raise ValueError('无法复现上轮T2验证池；请核对数据及原screen排除记录，不另选轨迹')
            result['normal_pools_disjoint_from_previous'] = True
            result['changed_validation_reused'] = True
        result['changed_pool_seed'] = changed_pool_seed
        pools = {**t1_pools, **t2_pools}
        result['episode_pools'] = {name:{**pool, 'dataset':names['probe'] if name=='changed_validation' else names['ref']}
                                   for name,pool in pools.items()}
        result['calibration_validation_episode_disjoint'] = True
        result['excluded_episode_counts'] = {r:len(set(e)) for r,e in excluded.items()}
        versions = {n:p._version for n,p in model.named_parameters()}
        buffers = {n:b.detach().cpu().clone() for n,b in model.named_buffers()}
        print('DRIFT_CAL_POOLS_READY', json.dumps({k:len(v['episode_ids']) for k,v in pools.items()}), flush=True)
        if previous is not None:
            print('DRIFT_CAL_PREVIOUS_POOLS_CHECKED', json.dumps(dict(
                normal_pools_disjoint_from_previous=True, changed_validation_reused=True)), flush=True)

        def measure(pool_name, seed, windows, label):
            sampled = copy.copy(args)
            sampled.seed, sampled.windows = seed, windows
            role = 'probe' if pool_name=='changed_validation' else 'ref'
            lens = t2_lengths if role=='probe' else t1_lengths
            return task_pred_error(model,datasets[role],np.asarray(pools[pool_name]['episode_ordinals']),
                lens,sampled,device,action_scaler=scaler,
                progress_callback=lambda n,total:print(f'DRIFT_CAL_PROGRESS {label} {n}/{total}',flush=True))

        result['status'] = 'reference'
        save()
        reference = measure('reference',args.seed+100, args.reference_windows,'reference')
        if not math.isfinite(reference[0]) or reference[0] <= 0:
            raise ValueError('固定参考误差无效')
        result['reference'] = dict(mean=reference[0],std=reference[1],n=reference[2],seed=args.seed+100)
        result['status'] = 'calibrating'
        save()
        for i in range(args.calibration_batches):
            seed = args.seed+1000+i
            error = measure('calibration',seed,args.windows,f'calibration_{i+1}')
            score,_ = drift_decision(reference[0],error[0],args.tau)
            result['calibration_rows'].append(dict(batch=i+1,seed=seed,mean=error[0],std=error[1],n=error[2],s=score))
            save()
            print('DRIFT_CAL_BATCH',i+1,args.calibration_batches,'score',score,flush=True)
        # Freeze before any prediction on either validation pool or on changed data.
        calibration = calibrated_threshold([r['s'] for r in result['calibration_rows']], args.target_fpr)
        tau = calibration['tau']
        result.update(calibration=calibration,threshold_calibrated=True,status='validating')
        save()
        print('DRIFT_CAL_THRESHOLD_FROZEN',json.dumps(calibration),flush=True)
        for i in range(args.validation_batches):
            for condition,pool in (('same_condition','validation'),('changed_condition','changed_validation')):
                seed = args.seed+100000+i
                error = measure(pool,seed,args.windows,f'{condition}_{i+1}')
                score,triggered = drift_decision(reference[0],error[0],tau)
                _,old_triggered = drift_decision(reference[0],error[0],args.tau)
                row = dict(batch=i+1,condition=condition,seed=seed,mean=error[0],std=error[1],n=error[2],
                           s=score,triggered=triggered,legacy_triggered=old_triggered,tau=tau)
                result['validation_rows'].append(row)
                save()
                print('DRIFT_CAL_VALIDATION',json.dumps(row),flush=True)
        if (versions != {n:p._version for n,p in model.named_parameters()} or
                not all(torch.equal(b.detach().cpu(),buffers[n]) for n,b in model.named_buffers())):
            raise RuntimeError('模型参数或统计发生变化')
        result['model_unchanged'] = True
        result['groups'] = {}
        for condition in ('same_condition','changed_condition'):
            rows = [r for r in result['validation_rows'] if r['condition']==condition]
            scores = [r['s'] for r in rows]
            result['groups'][condition] = dict(count=len(rows),triggered_count=sum(r['triggered'] for r in rows),
                legacy_triggered_count=sum(r['legacy_triggered'] for r in rows),score_min=min(scores),score_max=max(scores))
        result['status'] = 'complete'
        save()
        print('DRIFT_CAL_DONE',json.dumps(dict(out=str(out),calibration=calibration,groups=result['groups'],
            model_unchanged=True,elapsed_seconds=result['elapsed_seconds'])),flush=True)
        return result
    except Exception as exc:
        result.update(status='failed',error=str(exc))
        save()
        print('DRIFT_CAL_FAILED',str(exc),flush=True)
        raise


def run_screen(args, swm, all_tasks, scaler, action_metadata, manifest):
    """Repeated same-condition / changed-condition diagnostics, not calibrated rates."""
    safe = args.policy.replace('/', '_').replace(chr(92), '_').replace('.pt', '')
    out = (args.out or Path(f'drift_screen_{args.sequence}_{safe}.json')).resolve()
    if out.exists():
        raise FileExistsError(f'输出已存在，请用新路径，保留旧记录：{out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(status='preparing', protocol='fixed_action_v2_screen', policy=args.policy,
        sequence=args.sequence, ref_task=args.ref_task, probe_task=args.probe_task,
        tau=args.tau, threshold_validated=False, decision_only=True,
        repeats=args.screen_repeats, windows=args.windows, batch=args.batch, stride=args.stride,
        seed=args.seed, action_normalization=action_metadata, rows=[],
        scope='Offline dataset-condition diagnostic; reused episode pools, independence from '
              'training may be unverified. No training, growth or online-delay measurement. '
              'Changed datasets may differ in scene/action/source as well as appearance. '
              'Control trigger counts are not independently calibrated false-alarm rates.')
    def save():
        record['elapsed_seconds'] = time.monotonic()-started
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save()
    print('DRIFT_SCREEN_START ' + json.dumps(dict(policy=args.policy, repeats=args.screen_repeats,
          windows_per_group=args.windows, tau=args.tau)), flush=True)
    try:
        datasets, selections, lengths = {}, {}, {}
        for role, task in (('ref', args.ref_task), ('probe', args.probe_task)):
            name = all_tasks[task]['dataset']
            local = swm.data.utils.get_cache_dir(sub_folder='datasets') / name
            if not local.exists():
                raise FileNotFoundError(f'本地数据缺失，不自动下载：{local}')
            datasets[role] = swm.data.load_dataset(str(local), frameskip=1, num_steps=1,
                keys_to_load=['pixels', 'action'], keys_to_cache=['action'])
            selected, lens, meta = select_episodes(datasets[role], name, role, args.reserve_last, manifest)
            selections[role], lengths[role] = (selected, meta), lens
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = swm.data.utils.get_cache_dir(sub_folder='checkpoints') / args.policy
        if not checkpoint.is_file():
            raise FileNotFoundError(f'本地权重缺失，不自动下载：{checkpoint}')
        model = swm.wm.utils.load_pretrained(str(checkpoint)).to(device).eval()
        model.requires_grad_(False)
        span = int(getattr(model.predictor, 'num_frames', 3))*args.stride
        pools = {}
        for role, (selected, meta) in selections.items():
            pools[role] = np.asarray([e for e in selected if lengths[role][e] > span], dtype=selected.dtype)
            meta['excluded_short_episode_ids'] = [int(e) for e in selected if lengths[role][e] <= span]
            meta['episode_ids'] = pools[role].tolist()
            meta['minimum_required_raw_frames'] = span+1
            if not len(pools[role]):
                raise ValueError(f'{role}没有足够长的预测窗口')
        record['data_selection'] = {r:m for r,(_,m) in selections.items()}
        versions = {n:p._version for n,p in model.named_parameters()}
        buffers = {n:b.detach().cpu().clone() for n,b in model.named_buffers()}
        record['status'] = 'running'
        save()
        print('DRIFT_SCREEN_DATA_READY ' + json.dumps({r:len(p) for r,p in pools.items()}), flush=True)

        def measure(role, seed, label):
            sampled = copy.copy(args)
            sampled.seed = seed
            print('DRIFT_SCREEN_GROUP ' + json.dumps(dict(group=label, seed=seed)), flush=True)
            return task_pred_error(model, datasets[role], pools[role], lengths[role], sampled,
                device, action_scaler=scaler,
                progress_callback=lambda n,total: print(f'DRIFT_SCREEN_PROGRESS {label} {n}/{total}', flush=True))

        for repeat in range(args.screen_repeats):
            ref_seed = 2*(args.seed+repeat)
            probe_seed = ref_seed+1  # Different window sample; never an identical control replay.
            ref = measure('ref', ref_seed, f'ref_{repeat+1}')
            for condition, role in (('same_condition', 'ref'), ('changed_condition', 'probe')):
                probe = measure(role, probe_seed, f'{condition}_{repeat+1}')
                score, triggered = drift_decision(ref[0], probe[0], args.tau)
                row = dict(repeat=repeat+1, condition=condition, ref_seed=ref_seed, probe_seed=probe_seed,
                    err_ref=ref[0], std_ref=ref[1], n_ref=ref[2], err_probe=probe[0],
                    std_probe=probe[1], n_probe=probe[2], s=score, triggered=triggered)
                record['rows'].append(row)
                save()
                print('DRIFT_SCREEN_CASE ' + json.dumps(row, allow_nan=False), flush=True)
        if (versions != {n:p._version for n,p in model.named_parameters()} or
                not all(torch.equal(b.detach().cpu(), buffers[n]) for n,b in model.named_buffers())):
            raise RuntimeError('检测期间模型参数或统计发生变化')
        record['model_unchanged'] = True
        record['groups'] = {}
        for condition in ('same_condition', 'changed_condition'):
            rows = [r for r in record['rows'] if r['condition'] == condition]
            scores = [r['s'] for r in rows]
            record['groups'][condition] = dict(count=len(rows), triggered_count=sum(r['triggered'] for r in rows),
                score_mean=float(np.mean(scores)), score_min=min(scores), score_max=max(scores))
        record['status'] = 'complete'
        save()
        print('DRIFT_SCREEN_DONE ' + json.dumps(dict(out=str(out), groups=record['groups'],
            model_unchanged=True, threshold_validated=False, elapsed_seconds=record['elapsed_seconds'])), flush=True)
        return record
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        save()
        print('DRIFT_SCREEN_FAILED ' + str(exc), flush=True)
        raise


if __name__ == '__main__':
    main()
