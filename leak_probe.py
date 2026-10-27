"""显存泄漏定位探针：复刻 learn() 循环，按阶段打印 allocated/reserved。

跑 3 个迭代（rollout 250 步 × 100 envs + train 10 epochs），
定位泄漏发生在 rollout 阶段还是 train 阶段（含 minibatch 粒度）。
"""
import os
import sys
import gc

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import torch as th

from racing_env import make_racing_vec_env
from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.ppo_glue import ACMPCPPO
from stable_baselines3.common.callbacks import BaseCallback


def mem(tag):
    s = th.cuda.memory_stats()
    print(f"[mem] {tag}: alloc={th.cuda.memory_allocated()/2**30:.3f}GiB "
          f"reserved={th.cuda.memory_reserved()/2**30:.3f}GiB "
          f"retries={s.get('num_alloc_retries', '?')} ooms={s.get('num_ooms', '?')}", flush=True)


class NoopCallback(BaseCallback):
    def _on_step(self):
        return True


def main():
    env = make_racing_vec_env(track="horizontal", n_envs=100,
                              start_noise=0.5, normalize_obs=True)
    mem("after env")

    model = ACMPCPPO(
        policy=MlpMpcPolicy,
        env=env,
        learning_rate=lambda p: 3e-4 * (1 - p) + 1e-5 * p,
        n_steps=250, batch_size=25000, n_epochs=10,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        seed=0, verbose=0, device="cuda",
    )
    mem("after model")

    cb = NoopCallback()
    total_timesteps, _ = model._setup_learn(3 * 250 * 100, callback=cb)

    # minibatch 粒度计数：包一层 evaluate_actions
    counter = [0]
    orig_ea = model.policy.evaluate_actions

    def wrapped_ea(*a, **k):
        counter[0] += 1
        if counter[0] % 49 == 0:  # 每半个 epoch（98 个 minibatch 一轮）
            mem(f"  minibatch {counter[0]}")
        return orig_ea(*a, **k)

    model.policy.evaluate_actions = wrapped_ea

    # rollout 粒度计数：包一层 step_wait
    step_counter = [0]
    orig_step_wait = env.step_wait

    def wrapped_step_wait():
        step_counter[0] += 1
        if step_counter[0] % 50 == 0:
            mem(f"  rollout step {step_counter[0]}")
        return orig_step_wait()

    env.step_wait = wrapped_step_wait

    for it in range(3):
        model.collect_rollouts(model.env, cb, model.rollout_buffer, 250)
        mem(f"after rollout {it}")
        model.train()
        mem(f"after train {it}")
        gc.collect()
        mem(f"after gc {it}")

    print("LEAK_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
