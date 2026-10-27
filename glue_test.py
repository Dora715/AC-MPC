"""步骤 6 验证：ACMPCPPO 胶水补丁 + MlpMpcPolicy 的最小训练循环。

在 dummy 环境（36 维观测、前 10 维为悬停态 [p,q,v]）上跑 256 步 PPO，
覆盖 collect_rollouts 与 train 两个补丁调用点。通过标准：无 TypeError、
损失有值、GLUE_TEST_PASSED 打印。
"""
import os
import sys

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.env_util import make_vec_env

from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.ppo_glue import ACMPCPPO


class DummyRacingEnv(gym.Env):
    """36 维观测、4 维动作的占位环境；观测前 10 维是悬停态 [p, q, v]（q_w=1）。"""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        obs = np.zeros(36, dtype=np.float32)
        obs[3] = 1.0  # q_w = 1
        return obs

    def step(self, action):
        self.step_count += 1
        obs = np.zeros(36, dtype=np.float32)
        obs[3] = 1.0
        obs[9:36] = np.random.uniform(-0.1, 0.1, 27).astype(np.float32)
        done = self.step_count >= 100
        return obs, 0.0, done, {}


def main():
    env = make_vec_env(DummyRacingEnv, n_envs=2)

    model = ACMPCPPO(
        policy=MlpMpcPolicy,
        env=env,
        learning_rate=lambda _: 3e-4,
        n_steps=32,
        batch_size=64,
        n_epochs=2,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        verbose=1,
        device="cuda",
    )

    model.learn(total_timesteps=256)
    print("GLUE_TEST_PASSED")


if __name__ == "__main__":
    main()
