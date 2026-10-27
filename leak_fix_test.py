"""验证 closure 真空：train() 后 detach LQRStepFn 闭包 cell 里的张量。

预期：内存回到基线；下一 minibatch 梯度照常（真空不改求解器语义）。
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


def _sanitize(v):
    """把 cell 内容换成无图版本。"""
    if isinstance(v, th.Tensor):
        if v.is_cuda and v.requires_grad:
            return v.detach()
        return v
    if isinstance(v, tuple) and hasattr(v, "_fields"):
        try:
            return v._replace(
                **{f: _sanitize(getattr(v, f)) for f in v._fields})
        except Exception:
            return v
    if isinstance(v, list):
        return [_sanitize(x) for x in v]
    return v


def vacuum():
    """detach 所有 LQRStepFn 相关闭包 cell 中的图张量。"""
    n_cls = 0
    n_cells = 0
    for obj in gc.get_objects():
        try:
            if not (isinstance(obj, type) and obj.__name__ == "LQRStepFn"):
                continue
            if not issubclass(obj, th.autograd.Function):
                continue
            n_cls += 1
            for name in dir(obj):
                try:
                    fn = getattr(obj, name)
                except Exception:
                    continue
                if isinstance(fn, (staticmethod, classmethod)):
                    fn = fn.__func__
                if not callable(fn) or getattr(fn, "__closure__", None) is None:
                    continue
                for cell in fn.__closure__:
                    try:
                        v = cell.cell_contents
                    except ValueError:
                        continue
                    nv = _sanitize(v)
                    if nv is not v:
                        cell.cell_contents = nv
                        n_cells += 1
        except Exception:
            pass
    return n_cls, n_cells


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

    def minibatch(tag):
        m0 = mem()
        values, log_prob, entropy = policy.evaluate_actions(obs, actions, states)
        loss = (log_prob * adv).mean() + 0.5 * values.mean()
        loss.backward()
        m1 = mem()
        print(f"{tag}: {m0:.1f} -> {m1:.1f} (+{m1 - m0:+.1f}) "
              f"policy_net_grad={grad_norm(ext.policy_net):.4f}", flush=True)

    minibatch("mb1")
    minibatch("mb2")

    m_pre = mem()
    n_cls, n_cells = vacuum()
    ext.u_prev = None if ext.u_prev is None else ext.u_prev.detach()
    ext.predictions = None
    gc.collect()
    th.cuda.empty_cache()
    m_post = mem()
    print(f"vacuum: {n_cls} classes, {n_cells} cells detached; "
          f"mem {m_pre:.1f} -> {m_post:.1f} (freed {m_pre - m_post:+.1f})", flush=True)

    minibatch("mb3_after_vacuum")
    minibatch("mb4_after_vacuum")

    m_pre = mem()
    vacuum()
    gc.collect()
    th.cuda.empty_cache()
    m_post = mem()
    print(f"vacuum2: mem {m_pre:.1f} -> {m_post:.1f} (freed {m_pre - m_post:+.1f})", flush=True)
    print("VACUUM_DONE", flush=True)


if __name__ == "__main__":
    main()
