"""ACMPCPPO save/load 往返测试：验证 env + Welford 统计可 pickle、resume 可用。

这是分段训练（防 GPU 泄漏）的前提：每段结束 save 退出，新进程 load 续训。
检查点：保存能否 pickle（env/DroneDx/il_env/lr_schedule）、加载后
num_timesteps/Welford 统计/log_std 是否一致、续训是否正常。
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import numpy as np
import cloudpickle

from racing_env import make_racing_vec_env
from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.ppo_glue import ACMPCPPO

SAVE_DIR = "/root/acmpc/runs/ckpt/_roundtrip_test"


def lr_sched(p):
    # 模块级函数而非 lambda：普通 pickle 也可序列化
    return 3e-4 * p + 1e-5 * (1.0 - p)


def make_model(n_envs, seed):
    env = make_racing_vec_env(track="horizontal", n_envs=n_envs, start_noise=0.5, normalize_obs=True)
    return ACMPCPPO(
        policy=MlpMpcPolicy,
        env=env,
        learning_rate=lr_sched,
        n_steps=250,
        batch_size=1000,
        n_epochs=2,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        seed=seed,
        device="cuda",
    )


def gpu_used_mib():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().split("\n")[0])


def snap(model):
    return {
        "num_timesteps": model.num_timesteps,
        "mean": np.mean(np.stack(model.env.get_attr("mean")), axis=0),
        "count": float(np.mean(model.env.get_attr("count"))),
        "log_std": {k: v.detach().cpu().numpy() for k, v in model.policy.named_parameters() if "log_std" in k},
        "lr_fn": model.lr_schedule(0.5),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "model")

    t0 = time.time()
    m = make_model(n_envs=8, seed=0)
    m.learn(total_timesteps=2000, reset_num_timesteps=False)
    print(f"[A] learn 2000 ok in {time.time()-t0:.0f}s, num_timesteps={m.num_timesteps}", flush=True)
    a = snap(m)
    print(f"[A] count={a['count']:.1f} mean[0]={a['mean'][0]:.4f} log_std={a['log_std']} lr(0.5)={a['lr_fn']:.2e}", flush=True)

    t0 = time.time()
    m.save(path)
    with open(path + "_env.pkl", "wb") as f:
        cloudpickle.dump(m.env, f)
    print(f"[B] save ok in {time.time()-t0:.1f}s, zip={os.path.getsize(path + '.zip')/1e6:.1f} MB, "
          f"env_pkl={os.path.getsize(path + '_env.pkl')/1e6:.1f} MB, gpu_used={gpu_used_mib()} MiB", flush=True)

    del m
    import gc
    gc.collect()

    t0 = time.time()
    with open(path + "_env.pkl", "rb") as f:
        env2 = cloudpickle.load(f)
    m2 = ACMPCPPO.load(path, env=env2, device="cuda")
    print(f"[C] load ok in {time.time()-t0:.1f}s, gpu_used={gpu_used_mib()} MiB", flush=True)
    c = snap(m2)
    print(f"[C] num_timesteps={c['num_timesteps']} count={c['count']:.1f} "
          f"mean[0]={c['mean'][0]:.4f} log_std={c['log_std']} lr(0.5)={c['lr_fn']:.2e}", flush=True)

    assert c["num_timesteps"] == a["num_timesteps"], "num_timesteps mismatch"
    assert np.allclose(c["mean"], a["mean"]), "welford mean mismatch"
    assert abs(c["count"] - a["count"]) < 1e-6, "welford count mismatch"
    assert all(np.allclose(c["log_std"][k], a["log_std"][k]) for k in c["log_std"]), "log_std mismatch"
    assert abs(c["lr_fn"] - a["lr_fn"]) < 1e-9, "lr_schedule mismatch"
    print("[C] snapshot checks PASSED", flush=True)

    m2.learn(total_timesteps=1000, reset_num_timesteps=False)
    print(f"[D] continue learn ok, num_timesteps={m2.num_timesteps}", flush=True)
    # learn 以 rollout(n_steps×n_envs=2000) 为粒度，1000 步请求会按 1 个 rollout 取整 → 4000
    assert m2.num_timesteps == 4000, "resume counter broken"

    m2.env.env_method("_set_stats", a["mean"], a["count"], np.zeros_like(a["mean"]))
    print("[E] env_method _set_stats ok", flush=True)
    print("ROUNDTRIP_DONE", flush=True)


if __name__ == "__main__":
    main()
