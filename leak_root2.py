"""找根：minibatch 后哪些 requires_grad 的 CUDA 张量仍被引用。

泄漏 = 每 minibatch ~10 MiB（2 chunks）被永久保留。u_prev 只解释 1 MiB。
这里枚举 gc 中所有 requires_grad 的 cuda 张量（根），并 dump 状态快照。
"""
import collections
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

th.cuda.memory._record_memory_history(enabled="state", stacks="python")


def mem():
    return th.cuda.memory_allocated() / 2**20


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

    print(f"baseline: {mem():.1f} MiB", flush=True)

    def minibatch(tag):
        m0 = mem()
        values, log_prob, entropy = policy.evaluate_actions(obs, actions, states)
        loss = (log_prob * adv).mean() + 0.5 * values.mean()
        loss.backward()
        m1 = mem()
        print(f"{tag}: {m0:.1f} -> {m1:.1f} (+{m1 - m0:+.1f})", flush=True)

    minibatch("mb1")
    minibatch("mb2")

    th.cuda.memory._dump_snapshot("/root/acmpc/snap_root.pickle")

    gc.collect()
    roots = collections.Counter()
    big_tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, th.Tensor) and obj.is_cuda and obj.requires_grad:
                roots[tuple(obj.shape)] += 1
                if tuple(obj.shape) == (5, 1024, 14, 14):
                    big_tensors.append(obj)
        except Exception:
            pass
    print("LIVE REQUIRES_GRAD CUDA TENSORS:", flush=True)
    for shape, cnt in roots.most_common(40):
        print(f"  {shape}: {cnt}", flush=True)

    print("REFERRERS OF (5,1024,14,14) TENSORS:", flush=True)
    for t in big_tensors:
        print(f"  tensor {id(t)} grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}", flush=True)
        for ref in gc.get_referrers(t):
            desc = type(ref).__name__
            try:
                if isinstance(ref, th.autograd.function.BackwardCFunction):
                    desc += f" (ctx of {type(ref)}), saved={len(ref._saved_self._saved_tensors) if hasattr(ref, '_saved_self') else '?'}"
                elif isinstance(ref, th.Tensor):
                    desc += f" shape={tuple(ref.shape)}"
                elif isinstance(ref, list):
                    desc += f" list[{len(ref)}]"
                elif isinstance(ref, dict):
                    desc += f" dict keys={list(ref.keys())[:8]}"
                else:
                    desc += f" {str(ref)[:120]}"
            except Exception:
                pass
            print(f"    <- {desc}", flush=True)
    print("ROOT_DONE", flush=True)


if __name__ == "__main__":
    main()
