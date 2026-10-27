"""第二级探针：minibatch 粒度显存打印 + 第 2 个 minibatch 的分配栈快照。

用 torch.cuda.memory._record_memory_history(stacks="python") 记录第 2 个
minibatch 期间的所有分配，结束后 dump 快照到 snap.pickle，本地离线分析
存活块按调用栈聚合即可定位泄漏点。
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


class NoopCallback(BaseCallback):
    def _on_step(self):
        return True


def mem(tag):
    print(f"[mem] {tag}: alloc={th.cuda.memory_allocated()/2**30:.3f}GiB "
          f"reserved={th.cuda.memory_reserved()/2**30:.3f}GiB", flush=True)


def main():
    env = make_racing_vec_env(track="horizontal", n_envs=100,
                              start_noise=0.5, normalize_obs=True)
    model = ACMPCPPO(
        policy=MlpMpcPolicy,
        env=env,
        learning_rate=lambda p: 3e-4 * p + 1e-5 * (1.0 - p),
        n_steps=250, batch_size=25000, n_epochs=10,
        gamma=0.98, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.001, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        seed=0, verbose=0, device="cuda",
    )
    _total, cb = model._setup_learn(250 * 100, callback=NoopCallback())

    state = {"counter": 0}
    orig_ea = model.policy.evaluate_actions

    def wrapped_ea(*a, **k):
        state["counter"] += 1
        c = state["counter"]
        if c == 2:
            th.cuda.memory._record_memory_history(enabled="state", stacks="python")
            print("[hist] recording ON", flush=True)
        elif c == 3:
            th.cuda.memory._record_memory_history(enabled=None)
            th.cuda.memory._dump_snapshot("/root/acmpc/snap.pickle")
            print("[hist] recording OFF, snapshot dumped", flush=True)
        mem(f"  minibatch {c}")
        return orig_ea(*a, **k)

    model.policy.evaluate_actions = wrapped_ea

    model.collect_rollouts(model.env, cb, model.rollout_buffer, 250)
    mem("after rollout 0")
    model.train()
    mem("after train 0")
    gc.collect()
    mem("after gc")
    print("LEAK_PROBE2_DONE", flush=True)


if __name__ == "__main__":
    main()
