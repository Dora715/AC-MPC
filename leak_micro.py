"""最小复现：隔离泄漏机制。

A: forward_actor 循环（grad 模式，模拟训练路径）
B: il_env.mpc 单独循环（grad 模式）
C: il_env.mpc 单独循环（no_grad 模式）
D: 计时对比（图完整时的 forward+backward vs 纯 forward）
"""
import os
import sys

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import time
import numpy as np
import torch as th
from gym import spaces

from training_modules.mlp_mpc_policy import MlpMpcPolicy


def mem(tag):
    print(f"[mem] {tag}: alloc={th.cuda.memory_allocated()/2**20:.1f}MiB", flush=True)


def main():
    policy = MlpMpcPolicy(
        spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        lambda _: 3e-4,
    )
    policy.to("cuda")
    policy.set_training_mode(False)

    n = 1024
    obs = th.randn(n, 36, device="cuda")
    states = th.randn(n, 13, device="cuda")
    mem("setup")

    # A: forward_actor 循环（grad 模式）
    for i in range(3):
        out = policy.mlp_extractor.forward_actor(obs, states)
        mem(f"A grad forward {i}")

    # B: mpc 单独循环（grad 模式）
    dx = policy.mlp_extractor.dx
    il = policy.mlp_extractor.mpc_env
    Q = th.zeros(5, n, 14, 14, device="cuda")
    p = th.zeros(5, n, 14, device="cuda")
    Q[:, :, 0, 0] = 1.0
    for i in range(3):
        x_mpc, u_mpc = il.mpc(dx, states[:, :10].contiguous(), Q, p,
                              lqr_iter_override=1)
        mem(f"B grad mpc {i}")

    # C: mpc 单独循环（no_grad 模式）
    with th.no_grad():
        for i in range(3):
            x_mpc, u_mpc = il.mpc(dx, states[:, :10].contiguous(), Q, p,
                                  lqr_iter_override=1)
            mem(f"C nograd mpc {i}")

    # D: 计时（图完整 + backward）
    def run_backward():
        with th.enable_grad():
            Qd = th.zeros(5, n, 14, 14, device="cuda")
            pd_ = th.zeros(5, n, 14, device="cuda")
            Qd[:, :, 0, 0] = 1.0
            x_mpc, u_mpc = il.mpc(dx, states[:, :10].contiguous(), Qd, pd_,
                                  lqr_iter_override=1)
            loss = (x_mpc * x_mpc).sum() + (u_mpc * u_mpc).sum()
            loss.backward()

    th.cuda.synchronize()
    t0 = time.time()
    run_backward()
    th.cuda.synchronize()
    print(f"[time] D forward+backward: {time.time()-t0:.2f}s", flush=True)

    def run_fwd_only():
        with th.no_grad():
            x_mpc, u_mpc = il.mpc(dx, states[:, :10].contiguous(), Q, p,
                                  lqr_iter_override=1)

    th.cuda.synchronize()
    t0 = time.time()
    run_fwd_only()
    th.cuda.synchronize()
    print(f"[time] D forward only: {time.time()-t0:.2f}s", flush=True)

    print("MICRO_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
