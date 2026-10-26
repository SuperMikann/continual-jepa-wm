"""Render a few Push-T episodes and save as GIF.

Usage:
    python render_pusht.py --num_episodes 3 --max_steps 50 --output results/figures/pusht_demo.gif

Requires:
    pip install -e path/to/stable-worldmodel[env]
    pip install imageio pillow
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np


def render_episode(env, max_steps: int, seed: int | None = None):
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    frames = [env.render()]
    total_reward = 0.0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    return frames, total_reward, info


def main():
    parser = argparse.ArgumentParser(description='Render Push-T episodes')
    parser.add_argument('--env', default='swm/PushT-v1', help='Environment ID')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to render')
    parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output', default='results/figures/pusht_demo.gif', help='Output GIF path')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, render_mode='rgb_array')

    all_frames = []
    for ep in range(args.num_episodes):
        print(f'Rendering episode {ep + 1}/{args.num_episodes}...')
        frames, reward, info = render_episode(env, args.max_steps, seed=args.seed + ep)
        print(f'  Episode {ep + 1}: {len(frames)} frames, reward={reward:.3f}')

        # Add a few duplicate frames at episode boundary so GIF pauses
        all_frames.extend(frames)
        all_frames.extend([frames[-1]] * 5)

    env.close()

    # Convert frames to uint8 if needed
    all_frames = [
        (frame * 255).astype(np.uint8) if frame.dtype == np.float32 or frame.max() <= 1.0 else frame.astype(np.uint8)
        for frame in all_frames
    ]

    imageio.mimsave(output_path, all_frames, duration=0.1, loop=0)
    print(f'Saved GIF: {output_path}')


if __name__ == '__main__':
    main()
