"""决定性实验：u_prev 是否是每 minibatch 图保留的唯一持有者。

1. mb1: evaluate_actions + backward -> mem, policy_net grad 范数
2. mb2: 同样 -> 是否累积
3. 清空 u_prev/predictions + gc -> 是否释放
4. 打补丁（forward_actor 后 detach u_prev）后 mb3 -> 内存回落 + 梯度仍在
"""
import gc
import os
import sys

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")
sys.path.insert(0, "/root/acmpc/acmpc_public/diff_mpc_drones")

import numpy as np
import torch as th
from gym import spaces

from training_modules.mlp_mpc_policy import MlpMpcPolicy


def mem():
    return th.cuda.memory_allocated() / 2**20


def grad_norm(m):
    s = 0.0
    for p in m.parameters():
        if p.grad is not None:
            s += float(p.grad.norm().item() ** 2)
    return s ** 0.5


def main():
    policy = MlpMpcPolicy(
        spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        lambda _: 3e-4,
    )
    policy.to("cuda")
    policy.set_training_mode(True)
    ext = policy.mlp_extractor

    n = 2048
    obs = th.randn(n, 36, device="cuda")
    states = th.randn(n, 13, device="cuda")
    actions = th.randn(n, 4, device="cuda")
    adv = th.randn(n, 1, device="cuda")

    print(f"baseline: {mem():.1f} MiB", flush=True)

    def log_std_grad():
        for name, p in policy.named_parameters():
            if "log_std" in name and p.grad is not None:
                return float(p.grad.norm().item())
        return -1.0

    def minibatch(tag):
        m0 = mem()
        values, log_prob, entropy = policy.evaluate_actions(obs, actions, states)
        loss = (log_prob * adv).mean() + 0.5 * values.mean()
        loss.backward()
        m1 = mem()
        print(f"{tag}: before={m0:.1f} after_backward={m1:.1f} "
              f"growth={m1 - m0:+.1f} policy_net_grad={grad_norm(ext.policy_net):.4f} "
              f"log_std_grad={log_std_grad():.4f}", flush=True)
        return m1

    minibatch("mb1")
    minibatch("mb2")

    m_pre = mem()
    ext.u_prev = None
    ext.predictions = None
    gc.collect()
    th.cuda.empty_cache()
    m_post = mem()
    print(f"clear u_prev: {m_pre:.1f} -> {m_post:.1f} (freed {m_pre - m_post:+.1f})", flush=True)

    _orig = ext.forward_actor

    def patched(features, states_in):
        out = _orig(features, states_in)
        if ext.u_prev is not None:
            ext.u_prev = ext.u_prev.detach()
        return out

    ext.forward_actor = patched
    minibatch("mb3_patched")
    minibatch("mb4_patched")

    ext.forward_actor = _orig
    minibatch("mb5_restored")
    print("LEAK_ROOT_DONE", flush=True)


if __name__ == "__main__":
    main()
