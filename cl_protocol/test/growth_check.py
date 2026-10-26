"""Validate growth/freezing on discarded copies; never save trained weights.

Default: installed AdapterPredictor and scripts/train/lewm.py.
--module-file is for local CPU validation of the delivery source.
--policy additionally checks an existing real checkpoint, with random latent
inputs. This verifies implementation invariants, not prediction/control quality.
"""

import argparse
import ast
import copy
import importlib.util
import inspect
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class CheckEncoder(nn.Module):
    """CPU夹具：逐帧线性编码；其后使用真实LeWM和真实MLP/预测器。"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, pixels, **kwargs):
        return SimpleNamespace(last_hidden_state=self.linear(pixels).unsqueeze(1))


def load_symbols(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing repaired symbols in {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def check_mechanisms(module, world_model_file, train_file, eval_file):
    """检查实际修复函数；不启动Hydra训练、加载数据或生成真实实验成绩。"""
    from omegaconf import OmegaConf, open_dict
    from unittest.mock import patch
    spec = importlib.util.spec_from_file_location('mechanism_world_model', world_model_file)
    wm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wm
    spec.loader.exec_module(wm)
    helpers = load_symbols(train_file, ['gate_entropy_loss', 'lejepa_forward',
        'configure_mechanism', 'validate_mechanism', 'wrap_old_path_diagnostic', 'load_warm_start'],
        {'torch': torch, 'OmegaConf': OmegaConf, 'open_dict': open_dict})
    d = 12
    kw = dict(num_frames=3, input_dim=d, hidden_dim=d, depth=2, heads=2,
              dim_head=6, mlp_dim=24, dropout=0, emb_dropout=0)
    def make(predictor, legacy=False):
        return wm.LeWM(CheckEncoder(d), predictor, module.Embedder(2, 2, d),
            module.MLP(d, 24, d, norm_fn=nn.BatchNorm1d if legacy else nn.LayerNorm),
            module.MLP(d, 24, d, norm_fn=nn.BatchNorm1d) if legacy else nn.Identity())
    cfg = OmegaConf.create({'mechanism_version': 2,
        'wm': {'history_size': 3, 'num_preds': 1},
        'loss': {'sigreg': {'weight': 0}, 'delta_weight': .01, 'ent_weight': .1},
        'model': {'predictor': {'_target_': module.__name__ + '.AdapterPredictor'},
                  'projector': {'_target_': module.__name__ + '.MLP'},
                  'pred_proj': {'_target_': module.__name__ + '.MLP'}}})
    helpers['configure_mechanism'](cfg)
    assert cfg.model.pred_proj._target_ == 'torch.nn.Identity'
    assert cfg.model.projector.norm_fn._target_ == 'torch.nn.LayerNorm'
    assert cfg.model.predictor._target_.endswith('.AdapterPredictorV2')

    predictor = module.AdapterPredictorV2(**kw, num_adapters=2, rank=3, gate_hidden=8)
    model = make(predictor).train()
    helpers['validate_mechanism'](model, cfg)
    with torch.no_grad():
        predictor.adapter_A.normal_(std=.2)
        predictor.adapter_B.normal_(std=.2)
        predictor.delta_head.weight.normal_(std=.1)
        for block in predictor.transformer.layers:
            block.adaLN_modulation[-1].weight.normal_(std=.1)
    pixels, actions = torch.randn(2, 4, d), torch.randn(2, 4, 2)
    def full(model, pixels, actions):
        info = model.encode({'pixels': pixels, 'action': actions})
        return model.predict(info['emb'][:, :3], info['act_emb'][:, :3])
    first = full(model, pixels, actions)
    changed_pixels, changed_actions = pixels.clone(), actions.clone()
    changed_pixels[:, 1:] += torch.randn_like(changed_pixels[:, 1:]) * 5
    changed_actions[:, 1:] += torch.randn_like(changed_actions[:, 1:]) * 5
    second = full(model, changed_pixels, changed_actions)
    causal_diff = close_outputs(first[:, 0], second[:, 0])
    assert predictor.last_pi.shape == (2, 3, 2)
    grad_pixels = pixels.clone().requires_grad_(True)
    grad = torch.autograd.grad(full(model, grad_pixels, actions)[:, 0].sum(), grad_pixels)[0]
    assert torch.count_nonzero(grad[:, 1:]) == 0
    try:
        helpers['validate_mechanism'](make(predictor, legacy=True), cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('V2 accepted a post-residual projection/BatchNorm')
    bn_only = make(predictor)
    bn_only.projector = module.MLP(d, 24, d, norm_fn=nn.BatchNorm1d)
    try:
        helpers['validate_mechanism'](bn_only, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('V2 accepted temporal mixing through BatchNorm')

    logits = torch.tensor([[2., 0.]], requires_grad=True)
    pi = logits.softmax(-1)
    entropy = helpers['gate_entropy_loss'](pi, 2)
    gradient = torch.autograd.grad(entropy, logits)[0]
    entropy_after = helpers['gate_entropy_loss']((logits - .1 * gradient).softmax(-1), 2)
    assert entropy_after < entropy
    torch.testing.assert_close(helpers['gate_entropy_loss'](pi, 1), -entropy)

    diff_model = make(module.DiffPredictorV2(**kw)).eval()
    info = diff_model.encode({'pixels': pixels, 'action': actions})
    identity_diff = close_outputs(diff_model.predict(info['emb'][:, :3],
                        info['act_emb'][:, :3]), info['emb'][:, :3])
    trainer_module = SimpleNamespace(model=diff_model, sigreg=lambda z: z.sum() * 0,
                                    log_dict=lambda *a, **k: None)
    losses = helpers['lejepa_forward'](trainer_module,
        {'pixels': pixels, 'action': actions.clone()}, 'train', cfg)
    expected_l1 = (info['emb'][:, 1:] - info['emb'][:, :3]).abs().mean()
    torch.testing.assert_close(losses['delta_loss'], expected_l1)
    # 实际训练损失可反传，且缓存损失位于最终输出空间。
    losses['loss'].backward()
    assert diff_model.predictor.delta_head.weight.grad.abs().sum() > 0

    legacy = module.AdapterPredictor(**kw, num_adapters=2, rank=3, gate_hidden=8)
    try:
        predictor.load_state_dict(legacy.state_dict(), strict=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError('V2 silently loaded a legacy checkpoint')
    try:
        helpers['load_warm_start'](SimpleNamespace(predictor=legacy),
                                  {'predictor.mechanism_version': torch.tensor(2)})
    except ValueError:
        pass
    else:
        raise AssertionError('Legacy warm start silently accepted V2')
    restored_v2 = module.AdapterPredictorV2(**kw, num_adapters=2, rank=3, gate_hidden=8)
    restored_v2.load_state_dict(predictor.state_dict(), strict=True)
    x, c = torch.randn(2, 3, d), torch.randn(2, 3, d)
    v2_roundtrip = close_outputs(predictor.eval()(x, c), restored_v2.eval()(x, c))

    # 完整旧通路：实际LeWM + 含BN的历史投影 + 当前旧适配器冻结实现。
    base = make(module.AdapterPredictor(**kw, num_adapters=1, rank=3,
                        gate_hidden=8, freeze_gate=True), legacy=True).eval()
    with torch.no_grad():
        base.predictor.adapter_A.normal_(std=.1)
    fixed = module.FixedActionNormalizer([.2, -.3], [.7, 1.2])
    norm_actions = fixed.transform(actions)
    with torch.no_grad():
        old_prediction = full(base, pixels, norm_actions).clone()
    base.predictor.grow_adapter()
    diagnostic = module.FrozenOldPathWorldModel(base, 1, [.2, -.3], [.7, 1.2], 'CPU fixture')
    original_mean = diagnostic.action_mean.clone()
    diagnostic.to(dtype=torch.float32)
    assert diagnostic.action_mean.dtype == torch.float64
    assert torch.equal(original_mean, diagnostic.action_mean)
    saved_before = {k: v.clone() for k, v in diagnostic.state_dict().items()}
    diagnostic.set_condition('old')
    zero_diff = close_outputs(full(diagnostic, pixels, norm_actions), old_prediction)
    # 以实际原生rollout算固定候选和目标的规划成本，覆盖最终预测路径。
    candidates = torch.randn(2, 4, 5, 2)
    goal = pixels[:, -1:]
    def costs():
        raw_info = {'pixels': pixels[:, None, :3].expand(-1, 4, -1, -1),
                    'action_history': norm_actions[:, None, :2].expand(-1, 4, -1, -1)}
        rolled = diagnostic.rollout(raw_info, candidates)['predicted_emb']
        goal_emb = diagnostic.encode({'pixels': goal})['emb']
        return (rolled[:, :, -1] - goal_emb[:, None, -1]).square().sum(-1)
    with torch.no_grad():
        old_cost = costs().clone()
    optimizer = torch.optim.AdamW(diagnostic.parameters(), lr=.01, weight_decay=.1)
    handles = diagnostic.predictor.bind_frozen_optimizer(optimizer)
    diagnostic.train()
    assert not any(m.training for m in diagnostic.base.modules())
    diagnostic.set_condition('new')
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = full(diagnostic, pixels, norm_actions)
        (pred - old_prediction.detach() - .5).square().mean().backward()
        optimizer.step()
    for h in handles:
        h.remove()
    new_diff = float((full(diagnostic, pixels, norm_actions) - old_prediction).abs().max().detach())
    assert new_diff > 1e-6
    for key, value in diagnostic.state_dict().items():
        if key.endswith(('adapter_A', 'adapter_B')):
            assert torch.equal(value[:1], saved_before[key][:1])
        else:
            assert torch.equal(value, saved_before[key]), key
    diagnostic.set_condition('old')
    old_diff = close_outputs(full(diagnostic, pixels, norm_actions), old_prediction)
    cost_diff = close_outputs(costs(), old_cost)
    # 构造函数/保存后的实际加载能恢复同一已知条件通路。
    new_base = make(module.AdapterPredictor(**kw, num_adapters=2, rank=3,
                    gate_hidden=8, freeze_gate=True), legacy=True)
    restored = module.FrozenOldPathWorldModel(new_base, 1, [.2, -.3], [.7, 1.2], 'CPU fixture')
    restored.load_state_dict(diagnostic.state_dict(), strict=True)
    restored.set_condition('old')
    frozen_roundtrip = close_outputs(full(restored, pixels, norm_actions), old_prediction)
    restored.set_condition('new')
    diagnostic.set_condition('new')
    close_outputs(full(restored, pixels, norm_actions), full(diagnostic, pixels, norm_actions))
    torch.testing.assert_close(fixed.inverse_transform(norm_actions), actions)

    # 实际评估构造处必须用权重中的动作坐标，而不是当前评估集重拟合。
    captured = {}
    native = SimpleNamespace(wm=SimpleNamespace(utils=SimpleNamespace(load_pretrained=lambda *a: restored)),
        policy=SimpleNamespace(WorldModelPolicy=lambda **kw: captured.update(kw)))
    ev = load_symbols(eval_file, ['install_model_protocol'], {'swm': native, 'json': json, 'Path': Path})
    args = SimpleNamespace(condition_branch='old', action_stats=None)
    restore, receipt = ev['install_model_protocol'](args)
    try:
        native.wm.utils.load_pretrained('fixture')
        native.policy.WorldModelPolicy(process={'action': 'wrong per-task scaler'})
        torch.testing.assert_close(captured['process']['action'].transform(actions), norm_actions)
        assert restored.condition == 'old'
    finally:
        restore()
    assert receipt['normalization']['embedded_in_checkpoint']
    # 用实际训练包装函数生成可保存配置，再由Hydra重建并严格加载状态。
    from hydra.utils import instantiate
    mlp_cfg = {'_target_': module.__name__ + '.MLP', 'input_dim': d, 'hidden_dim': 24,
               'output_dim': d, 'norm_fn': {'_target_': 'torch.nn.BatchNorm1d', '_partial_': True}}
    base_cfg = {'_target_': spec.name + '.LeWM',
        'encoder': {'_target_': CheckEncoder.__module__ + '.CheckEncoder', 'dim': d},
        'predictor': {'_target_': module.__name__ + '.AdapterPredictor', **kw,
                      'num_adapters': 2, 'rank': 3, 'gate_hidden': 8, 'freeze_gate': True},
        'action_encoder': {'_target_': module.__name__ + '.Embedder',
                           'input_dim': 2, 'smoothed_dim': 2, 'emb_dim': d},
        'projector': mlp_cfg, 'pred_proj': mlp_cfg}
    wrap_cfg = OmegaConf.create({'model': base_cfg,
        'fixed_action_normalization': {'mean': [.2, -.3], 'scale': [.7, 1.2], 'source': 'CPU fixture'}})
    with patch.dict(sys.modules, {'stable_worldmodel.wm.lewm.module': module}):
        wrapped = helpers['wrap_old_path_diagnostic'](instantiate(wrap_cfg.model), wrap_cfg, 1)
        wrapped.load_state_dict(diagnostic.state_dict(), strict=True)
        # 真实部署路径使用可导入的类名（CPU别名只用于本测试进程）。
        saved_cfg = OmegaConf.to_container(wrap_cfg.model, resolve=True)
        saved_cfg['_target_'] = module.__name__ + '.FrozenOldPathWorldModel'
        loaded = instantiate(OmegaConf.create(json.loads(json.dumps(saved_cfg))))
        loaded.load_state_dict(wrapped.state_dict(), strict=True)
        loaded.set_condition('old')
        close_outputs(full(loaded, pixels, norm_actions), old_prediction)
    return {'complete_path_future_change_max_abs': causal_diff,
            'future_input_gradient_nonzero_count': int(torch.count_nonzero(grad[:, 1:])),
            'entropy_before': float(entropy.detach()), 'entropy_after_descent': float(entropy_after.detach()),
            'full_residual_zero_initialization_max_abs': identity_diff,
            'training_delta_loss_in_final_space': True, 'training_backward_passed': True,
            'legacy_to_v2_rejected': True, 'v2_checkpoint_roundtrip_max_abs': v2_roundtrip,
            'old_path_zero_step_max_abs': zero_diff, 'old_path_after_3_adamw_steps_max_abs': old_diff,
            'old_planning_cost_after_steps_max_abs': cost_diff, 'new_path_max_change': new_diff,
            'all_shared_parameters_and_buffers_unchanged': True,
            'frozen_checkpoint_roundtrip_max_abs': frozen_roundtrip,
            'evaluation_fixed_action_normalizer_used': True,
            'fixed_action_statistics_preserved_by_model_cast': True,
            'training_generated_config_hydra_roundtrip_passed': True,
            'v2_to_legacy_warm_start_rejected': True,
            'scope': 'Real local model classes/helpers; synthetic inputs and CPU encoder fixture; no environment result'}


def load_training_helpers(path):
    # Execute precisely the helpers used by run(), without importing datasets,
    # initializing Hydra or launching the training entry point.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = {'load_warm_start', 'FreezeOldAdaptersCallback'}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError('Upload the repaired train_lewm.py before this check')
    namespace = {'Callback': object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    calls = {n.func.id for n in ast.walk(run)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    if not wanted <= calls:
        raise RuntimeError('Training run() does not use the validated loader/callback')
    return namespace['load_warm_start'], namespace['FreezeOldAdaptersCallback']


def close_outputs(a, b):
    difference = float((a - b).detach().abs().max())
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    return difference


def check_model(model, warm_start, callback_type, label, device):
    model = model.to(device).eval()
    predictor = model.predictor
    k = predictor.num_adapters
    if not predictor.freeze_gate:
        raise ValueError('This check targets uniform-routing growth')
    if not predictor.adapter_A.abs().max() > 0:
        raise ValueError('Old adapter A is zero: this would be a vacuous growth check')
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = torch.randn(2, predictor.num_frames, predictor.input_dim, device=device)
    c = torch.randn_like(x)
    with torch.no_grad():
        before = predictor(x, c).clone()
    # deepcopy before retaining a backward graph in any cached intermediates.
    grown = copy.deepcopy(model)
    grown.predictor.grow_adapter()
    with torch.no_grad():
        direct = grown.predictor(x, c).clone()
    direct_diff = close_outputs(before, direct)
    assert torch.equal(grown.predictor.adapter_A[:k], predictor.adapter_A)
    assert torch.equal(grown.predictor.adapter_B[:k], predictor.adapter_B)

    # Real training path: instantiate target size, then warm-start its state.
    target = copy.deepcopy(grown)
    with torch.no_grad():
        target.predictor.adapter_A.zero_()
        target.predictor.adapter_B.normal_(std=0.01)
        target.predictor.adapter_scale.fill_(1)
    missing, unexpected = warm_start(target, original)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        warm = target.predictor(x, c).clone()
    warm_diff = close_outputs(before, warm)
    assert torch.equal(target.predictor.adapter_A[:k], predictor.adapter_A)
    assert torch.equal(target.predictor.adapter_B[:k], predictor.adapter_B)

    # Serialization must carry the correction; it cannot live only in RAM.
    stream = io.BytesIO()
    torch.save(target.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, map_location=device, weights_only=True)
    restored = copy.deepcopy(target)
    restored.predictor.adapter_scale.fill_(1)
    restored.load_state_dict(saved, strict=True)
    with torch.no_grad():
        reload_diff = close_outputs(warm, restored.predictor(x, c))

    # The actual training callback binds a real AdamW optimizer. Use evaluation
    # mode so dropout does not obscure the update check. Shared layers may move.
    target.predictor.freeze_old_adapters(k)
    optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5, weight_decay=0.001)
    trainer = SimpleNamespace(optimizers=[optimizer])
    wrapper = SimpleNamespace(model=target)
    callback = callback_type()
    callback.on_fit_start(trainer, wrapper)
    old_a = target.predictor.adapter_A[:k].detach().clone()
    old_b = target.predictor.adapter_B[:k].detach().clone()
    new_a = target.predictor.adapter_A[k:].detach().clone()
    old_gradient_max = 0.0
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        target.predictor(x, c).square().mean().backward()
        for p in (target.predictor.adapter_A, target.predictor.adapter_B):
            old_gradient_max = max(old_gradient_max, float(p.grad[:k].abs().max()))
            assert torch.count_nonzero(p.grad[:k]) == 0
            if step == 1:
                # Extra stress case: a resumed optimizer can have old momentum.
                optimizer.state[p]['exp_avg'][:k].fill_(0.7)
                optimizer.state[p]['exp_avg_sq'][:k].fill_(0.8)
        optimizer.step()
        assert torch.equal(old_a, target.predictor.adapter_A[:k])
        assert torch.equal(old_b, target.predictor.adapter_B[:k])
    new_change = float((target.predictor.adapter_A[k:] - new_a).detach().abs().max())
    assert new_change > 0, 'New adapter did not update'
    callback.on_fit_end(trainer, wrapper)
    assert not callback.handles
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    return dict(label=label, old_slots=k, new_slots=k+1,
                zero_step_growth_max_abs_diff=direct_diff,
                training_warm_start_max_abs_diff=warm_diff,
                checkpoint_roundtrip_max_abs_diff=reload_diff,
                old_gradient_max=old_gradient_max,
                old_parameter_max_change_after_steps=0.0,
                new_adapter_max_change=new_change,
                adamw_lr=5e-5, adamw_weight_decay=0.001,
                optimizer_steps_on_discarded_copy=2,
                second_step_includes_synthetic_old_momentum=True,
                original_in_memory_state_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-file', type=Path)
    parser.add_argument('--train-file', type=Path, default=Path('scripts/train/lewm.py'))
    parser.add_argument('--policy', help='Optional real checkpoint; read only')
    parser.add_argument('--world-model-file', type=Path,
                        help='Also validate repaired mechanisms using this actual LeWM source')
    parser.add_argument('--eval-file', type=Path, default=Path('cl_protocol/eval_task.py'))
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.device == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.module_file:
        spec = importlib.util.spec_from_file_location('repaired_growth_module', args.module_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        AdapterPredictor = module.AdapterPredictor
    else:
        from stable_worldmodel.wm.lewm.module import AdapterPredictor
    warm_start, callback = load_training_helpers(args.train_file)
    print('[GROWTH_CHECK_SOURCE]', inspect.getfile(AdapterPredictor), flush=True)
    kw = dict(num_frames=3, depth=2, heads=2, mlp_dim=64,
              input_dim=32, hidden_dim=32, dim_head=16,
              num_adapters=1, rank=4, gate_hidden=8, freeze_gate=True)
    source = nn.Module()
    source.predictor = AdapterPredictor(**kw)
    with torch.no_grad():
        source.predictor.adapter_A.normal_(std=0.05)
        source.predictor.adapter_B.normal_(std=0.1)
    # Legacy checkpoints have no scale key. Strict loading must still work.
    legacy = {k: v for k, v in source.state_dict().items() if not k.endswith('adapter_scale')}
    source.load_state_dict(legacy, strict=True)
    same_size = nn.Module()
    same_size.predictor = AdapterPredictor(**kw)
    missing, unexpected = warm_start(same_size, legacy)
    assert not missing and not unexpected
    assert all(torch.equal(value, same_size.state_dict()[key])
               for key, value in source.state_dict().items())
    expanded = nn.Module()
    expanded.predictor = AdapterPredictor(**{**kw, 'num_adapters': 2})
    missing, unexpected = warm_start(expanded, legacy)
    assert not missing and not unexpected
    assert float(expanded.predictor.adapter_scale) == 2
    with torch.no_grad():
        # Simulate a learned second branch before the next expansion.
        expanded.predictor.adapter_A[1].normal_(std=0.03)
    checks = [check_model(source, warm_start, callback, 'synthetic_K1_to_K2', args.device),
              check_model(expanded, warm_start, callback, 'synthetic_K2_to_K3', args.device)]
    free = AdapterPredictor(**{**kw, 'freeze_gate': False})
    for action in (lambda: free.grow_adapter(),
                   lambda: AdapterPredictor(**{**kw, 'num_adapters': 3, 'freeze_gate': False}).prepare_growth_state_dict(expanded.predictor.state_dict())):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported learned-gate growth was not rejected')
    if args.policy:
        if args.module_file:
            parser.error('Real checkpoint check must use the installed repaired module')
        import stable_worldmodel as swm
        real = swm.wm.utils.load_pretrained(args.policy)
        checks.append(check_model(real, warm_start, callback, 'real_checkpoint', args.device))
    result = dict(status='passed', checks=checks, policy=args.policy,
                  legacy_strict_load_passed=True, legacy_same_size_warm_start_passed=True,
                  legacy_expanding_warm_start_passed=True,
                  learned_gate_growth_rejected=True,
                  scope='Random latent inputs; no dataset training or behavior evaluation; no checkpoint file writes')
    if args.world_model_file:
        if not args.module_file:
            # wm.lewm can be a re-exported model module, not the package attribute.
            from importlib import import_module
            module = import_module('stable_worldmodel.wm.lewm.module')
        result['mechanism_repairs'] = check_mechanisms(
            module, args.world_model_file, args.train_file, args.eval_file
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('[GROWTH_CHECK_OK]', json.dumps(result, ensure_ascii=False), flush=True)
    print('[GROWTH_CHECK_DONE]', args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
