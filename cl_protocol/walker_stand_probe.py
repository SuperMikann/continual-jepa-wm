"""Walker 直立保持探针：站姿起步 + 站立目标，测模型能维持直立多久。

与 qpos_match 评估的差异：不从数据集的不稳定中间状态起步，
而是 walker 默认站立初始化；目标 = 起步帧本身（"保持站立"）。
用随机射击（random shooting）规划，测直立时长与前进速度——
判断"模型 + GoalMSE 规划"在 locomotion 上是否完全不可用。

用法：python cl_protocol/walker_stand_probe.py \
  --policy m5_walker_t1/weights_epoch_30.pt --episodes 10
"""

import argparse
import numpy as np
import torch

import stable_worldmodel as swm
from drift_world import make_darken
from pred_error import img_transform


def cem_once(model, z_hist, a_hist, z_goal, act_dim, n_samples=512, n_iter=5, horizon=5, device='cuda'):
    """单步 CEM：从 z_hist 出发，horizon 步 rollout，按与 z_goal 的 MSE 选动作。"""
    best_mean, best_std = np.zeros((horizon, act_dim)), np.ones((horizon, act_dim))
    for _ in range(n_iter):
        acts = np.stack([np.clip(np.random.normal(best_mean[h], best_std[h], n_samples), -1, 1) for h in range(horizon)], 1)
        with torch.no_grad():
            z = z_hist.repeat(n_samples, 1, 1)
            a = torch.as_tensor(a_hist, device=device).repeat(n_samples, 1, 1)
            for h in range(horizon):
                a_new = torch.as_tensor(acts[:, h], device=device, dtype=torch.float32)
                a_step = model.action_encoder(a_new).unsqueeze(1)
                z = model.predictor(z, a_step)[:, -1:]
            cost = ((z[:, -1] - z_goal) ** 2).mean(-1)
        elite = acts[cost.argsort()[: max(1, n_samples // 20)]]
        best_mean, best_std = elite.mean(0), elite.std(0) + 1e-4
    return best_mean[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval().requires_grad_(False)
    tf = img_transform()

    heights, speeds, falls = [], [], 0
    for ep in range(args.episodes):
        world = swm.World('swm/WalkerDMControl-v0', num_envs=1,
                          image_shape=(224, 224), max_episode_steps=args.steps)
        obs, infos = world.reset(seed=ep)
        # 目标 = 起步帧（保持站立）
        goal_pix = tf(torch.as_tensor(infos['pixels'][0, 0]).permute(2, 0, 1)).unsqueeze(0).unsqueeze(0).to(device)
        z_goal = model.encode({'pixels': goal_pix})['emb'][0, -1]

        frames, hist_a = [torch.as_tensor(infos['pixels'][0, 0])], [np.zeros(infos['action'].shape[-1] * 5, dtype=np.float32)]
        h, spd = [], []
        for t in range(args.steps):
            hist_f = torch.stack([tf(f.permute(2, 0, 1)) for f in frames[-3:]]).unsqueeze(0).to(device)
            info = model.encode({'pixels': hist_f, 'action': torch.zeros(1, 3, 1, device=device)})
            a = cem_once(model, info['emb'], info['act_emb'], z_goal,
                         act_dim=world.action_space.shape[0], device=device)
            obs, infos, *_ = world.step(np.array([a])) if hasattr(world, 'step') else world.envs.step(np.array([a]))
            frames.append(torch.as_tensor(infos['pixels'][0, 0] if isinstance(infos, dict) else infos[1]['pixels'][0, 0]))
            h.append(float(infos['torso_height'][0]) if isinstance(infos, dict) else 0.0)
            spd.append(float(infos['speed'][0]) if isinstance(infos, dict) else 0.0)
        upright = sum(1 for x in h if x > 0.8)
        heights.append(upright / args.steps)
        speeds.append(np.mean(spd))
        print(f'ep{ep}: 直立比例 {upright}/{args.steps}, 平均速度 {np.mean(spd):.2f}')

    print(f'\n[结果] 直立时间比例 = {np.mean(heights):.2f} ± {np.std(heights):.2f}')
    print(f'[结果] 平均前进速度 = {np.mean(speeds):.2f} ± {np.std(speeds):.2f} m/s')


if __name__ == '__main__':
    main()
