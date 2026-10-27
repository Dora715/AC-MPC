"""梯度路径检查：单 minibatch evaluate_actions + backward，看 policy_net 是否有梯度。

同时测：完整梯度路径下（不回传时），各参数的 grad 范数。
"""
import os
import sys

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import numpy as np
import torch as th
from gym import spaces

from racing_env import make_racing_vec_env
from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.ppo_glue import ACMPCPPO


def main():
    env = make_racing_vec_env(track="horizontal", n_envs=100,
                              start_noise=0.5, normalize_obs=True)
    model = ACMPCPPO(
        policy=MlpMpcPolicy, env=env,
        learning_rate=lambda p: 3e-4 * p + 1e-5 * (1.0 - p),
        n_steps=250, batch_size=25000, n_epochs=10,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        seed=0, verbose=0, device="cuda",
    )
    from stable_baselines3.common.callbacks import BaseCallback
    class NoopCallback(BaseCallback):
        def _on_step(self):
            return True
    _t, cb = model._setup_learn(250 * 100, callback=NoopCallback())
    model.collect_rollouts(model.env, cb, model.rollout_buffer, 250)

    rollout_data = next(iter(model.rollout_buffer.get(model.batch_size)))
    model.policy.set_training_mode(True)

    values, log_prob, entropy = model.policy.evaluate_actions(
        rollout_data.observations, rollout_data.actions, rollout_data.states
    )

    print("values.requires_grad:", values.requires_grad,
          " grad_fn:", type(values.grad_fn).__name__ if values.grad_fn else None)
    print("log_prob.requires_grad:", log_prob.requires_grad,
          " grad_fn:", type(log_prob.grad_fn).__name__ if log_prob.grad_fn else None)

    loss = log_prob.mean() + values.mean()
    model.policy.optimizer.zero_grad()
    loss.backward()

    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            print(f"GRAD {name:55s} norm={param.grad.norm().item():.4e}")
        else:
            print(f"NOGRAD {name}")

    print("GRADCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
