"""识别 list[6] 结构与 C++ 侧持有者。"""
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


def desc(x):
    try:
        if isinstance(x, th.Tensor):
            return (f"Tensor{tuple(x.shape)} rg={x.requires_grad} "
                    f"gf={type(x.grad_fn).__name__ if x.grad_fn else None}")
        if isinstance(x, list):
            return f"list[{len(x)}]"
        if isinstance(x, tuple):
            return f"tuple[{len(x)}]"
        return f"{type(x).__name__}"
    except Exception:
        return type(x).__name__


def main():
    policy = MlpMpcPolicy(
        spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        lambda _: 3e-4,
    )
    policy.to("cuda")
    policy.set_training_mode(True)

    n = 2048
    obs = th.randn(n, 36, device="cuda")
    states = th.randn(n, 13, device="cuda")
    actions = th.randn(n, 4, device="cuda")
    adv = th.randn(n, 1, device="cuda")

    values, log_prob, entropy = policy.evaluate_actions(obs, actions, states)
    loss = (log_prob * adv).mean() + 0.5 * values.mean()
    loss.backward()
    loss = values = log_prob = entropy = None
    print(f"after mb1: {mem():.1f} MiB", flush=True)

    gc.collect()
    leaves = [
        t for t in gc.get_objects()
        if isinstance(t, th.Tensor) and t.is_cuda and t.requires_grad
        and tuple(t.shape) == (4096, 4)
    ]
    print(f"found {len(leaves)} leaves", flush=True)
    leaf = leaves[0]
    lists6 = [r for r in gc.get_referrers(leaf) if isinstance(r, list)]
    for l6 in lists6:
        print(f"list[6] contents: {[desc(x) for x in l6]}", flush=True)
        # 该 list 的引用者
        for ref in gc.get_referrers(l6):
            print(f"  list[6] <- {desc(ref)}", flush=True)
        # 也检查 tensor 的 grad 属性
    for t in leaves[:1]:
        g = getattr(t, "grad", None)
        print(f"leaf.grad: {desc(g) if g is not None else None}", flush=True)

    # 全图 scan: 有多少 LQRStepFn 类 / 多少 BackwardCFunction 对象
    n_cls = sum(
        1 for o in gc.get_objects()
        if isinstance(o, type) and o.__name__ == "LQRStepFn")
    n_bcf = sum(
        1 for o in gc.get_objects()
        if isinstance(o, th.autograd.function.BackwardCFunction))
    n_ctx = sum(
        1 for o in gc.get_objects()
        if type(o).__name__ in ("CudaFunctionCtx", "FunctionCtx"))
    print(f"LQRStepFn classes={n_cls} BackwardCFunction={n_bcf} ctx={n_ctx}", flush=True)
    print("STRUCT_DONE", flush=True)


if __name__ == "__main__":
    main()
