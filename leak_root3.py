"""从 _Q 沿引用链向上走，找到终极持有者。"""
import gc
import os
import sys
import types

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


def describe(obj):
    try:
        if isinstance(obj, th.Tensor):
            return (f"Tensor shape={tuple(obj.shape)} "
                    f"requires_grad={obj.requires_grad} "
                    f"grad_fn={type(obj.grad_fn).__name__ if obj.grad_fn else None} "
                    f"is_leaf={obj.is_leaf}")
        if isinstance(obj, list):
            return f"list[{len(obj)}]"
        if isinstance(obj, tuple):
            return f"tuple[{len(obj)}]"
        if isinstance(obj, dict):
            return f"dict keys={list(obj.keys())[:10]}"
        if isinstance(obj, th.autograd.function.BackwardCFunction):
            saved = getattr(getattr(obj, "_saved_self", None), "_saved_tensors", None)
            return f"BackwardCFunction {type(obj).__name__} saved={len(saved) if saved else None}"
        if isinstance(obj, th.nn.Module):
            return f"Module {type(obj).__name__}"
        if isinstance(obj, types.FrameType):
            return (f"FRAME {obj.f_code.co_filename}:{obj.f_lineno} "
                    f"in {obj.f_code.co_name} "
                    f"locals={sorted(obj.f_locals.keys())[:15]}")
        if isinstance(obj, types.FunctionType):
            return (f"FUNCTION {getattr(obj, '__name__', '?')} "
                    f"at {getattr(obj, '__code__', None) and obj.__code__.co_filename}:"
                    f"{getattr(obj, '__code__', None) and obj.__code__.co_firstlineno}")
        if isinstance(obj, types.GeneratorType):
            return f"GENERATOR of {obj.gi_code.co_filename}:{obj.gi_code.co_firstlineno} {obj.gi_code.co_name}"
        if isinstance(obj, types.CellType):
            return "cell"
        if hasattr(obj, "__dict__"):
            d = getattr(obj, "__dict__", {})
            keys = list(d.keys())[:12]
            return f"{type(obj).__name__} attrs={keys}"
        return f"{type(obj).__name__}"
    except Exception:
        return type(obj).__name__


def walk(obj, depth, seen):
    if depth > 14 or id(obj) in seen:
        return
    seen.add(id(obj))
    print(f"{'  ' * depth}[d{depth}] {describe(obj)}", flush=True)
    for ref in gc.get_referrers(obj):
        walk(ref, depth + 1, seen)


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
    print(f"after mb1: {mem():.1f} MiB", flush=True)

    gc.collect()
    targets = [
        t for t in gc.get_objects()
        if isinstance(t, th.Tensor) and t.is_cuda and t.requires_grad
        and tuple(t.shape) in ((5, 1024, 14, 14), (4096, 4), (5, 1024, 4))
    ]
    print(f"found {len(targets)} target tensors", flush=True)
    for t in targets[:1]:
        print(f"=== walking up from {tuple(t.shape)} {id(t)} ===", flush=True)
        walk(t, 0, set())
    print("WALK_DONE", flush=True)


if __name__ == "__main__":
    main()
