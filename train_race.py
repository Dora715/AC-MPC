"""步骤 8：AC-MPC (MlpMpcPolicy) vs AC-MLP (MlpOnlyPolicy) 训练脚本（分段+续训版）。

超参按论文 Table II：γ=0.98、GAE λ=0.95、n_steps=250、batch 25000（→ n_envs=100）、
10 epochs、clip 0.2、lr 3e-4→1e-5、entropy 0.001、vf 0.5、grad clip 0.5、
log_std 初值 −1.2（AC-MPC）/ −0.5（AC-MLP）。训练 2M 步。

分段训练（防 GPU 泄漏）：mpc.pytorch 的 LQRStep 工厂把每批 minibatch 的
_Q/_p/热启动张量钉死在 C++ 根引用上（约 1.19 GiB/训练，进程内不可释放，
见项目记录）。因此每训练 --segment-timesteps 步后保存模型+env 并以退出码
42 结束，由 run_all.sh 重启新进程续训（泄漏随进程退出归还 GPU）。
env 需单独 pickle（fork SB3 1.7 的 save 排除 env，而 Welford 归一化统计
存在 env 里），resume 时加载后传入 ACMPCPPO.load(env=...)。

用法（服务器 /root/acmpc 下）：
  python train_race.py --track horizontal --policy mpc  --total-timesteps 2000000 --tag run1
  python train_race.py --track horizontal --policy mpc  --total-timesteps 2000000 --tag run1 --resume <ckpt路径>
"""
import argparse
import os
import sys
import time

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import cloudpickle
import numpy as np
import torch as th

from racing_env import STATE_DIM, make_racing_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.mlp_only_policy import MlpOnlyPolicy
from training_modules.ppo_glue import ACMPCPPO

POLICIES = {"mpc": MlpMpcPolicy, "mlp": MlpOnlyPolicy}
LOG_STD = {"mpc": -1.2, "mlp": -0.5}

MAX_EPISODE_STEPS = 500
SEGMENT_EXIT_CODE = 42


def linear_lr(progress_remaining):
    # 模块级函数而非 lambda（普通 pickle 也可序列化）。
    # SB3 传入 progress_remaining（1→0）：3e-4 线性衰减到 1e-5。
    return 3e-4 * progress_remaining + 1e-5 * (1.0 - progress_remaining)


def save_ckpt(model, path):
    """保存模型 zip + env pickle（Welford 统计在 env 里，fork SB3 的 save 不含 env）。"""
    model.save(path)
    with open(path + "_env.pkl", "wb") as f:
        cloudpickle.dump(model.env, f)


class TrackEvalCallback(BaseCallback):
    """每 eval_freq 步用确定性策略在单环境上评估：过门数、完赛率。"""

    def __init__(self, track, eval_freq, n_eval_eps=10, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_eps = n_eval_eps
        self.eval_env = make_racing_vec_env(track=track, n_envs=1, start_noise=0.5, normalize_obs=True)

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        gates_list, success = [], 0
        # 评估环境复用训练环境的归一化统计，避免观测分布错配
        tr_mean = np.mean(np.stack(self.model.env.get_attr("mean")), axis=0)
        tr_count = float(np.mean(self.model.env.get_attr("count")))
        tr_m2 = np.mean(np.stack(self.model.env.get_attr("m2")), axis=0)
        self.eval_env.env_method("_set_stats", tr_mean, tr_count, tr_m2)
        with th.no_grad():
            for _ in range(self.n_eval_eps):
                obs, state = self.eval_env.reset()
                gates = 0
                for _ in range(MAX_EPISODE_STEPS):
                    obs_t = obs_as_tensor(obs, self.model.device)
                    st_t = th.as_tensor(state, dtype=th.float32).to(self.model.device)
                    actions, _, _ = self.model.policy(obs_t, st_t, deterministic=True)
                    act = np.clip(actions.cpu().numpy(), -1.0, 1.0)
                    obs, rews, dones, infos = self.eval_env.step(act)
                    if dones[0]:
                        gates = infos[0].get("gates_passed", 0)
                        if infos[0].get("is_success", False):
                            success += 1
                        break
                gates_list.append(gates)
        self.logger.record("eval/gates_passed", float(np.mean(gates_list)))
        self.logger.record("eval/success_rate", success / self.n_eval_eps)


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._last = 0

    def init_callback(self, model):
        super().init_callback(model)
        # resume 后从当前步数对齐，避免下个保存点偏移 1 步
        self._last = model.num_timesteps

    def _on_step(self):
        if self.num_timesteps >= self._last + self.save_freq:
            self._last = self.num_timesteps
            save_ckpt(self.model, os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}"))
        return True


def make_model(track, policy_name, n_envs, seed, log_dir, run_name):
    env = make_racing_vec_env(track=track, n_envs=n_envs, start_noise=0.5, normalize_obs=True)
    model = ACMPCPPO(
        policy=POLICIES[policy_name],
        env=env,
        learning_rate=linear_lr,
        n_steps=250,
        batch_size=25000,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"log_std_init": LOG_STD[policy_name]},
        seed=seed,
        tensorboard_log=os.path.join(log_dir, "tb"),
        verbose=1,
        device="cuda",
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="horizontal", choices=["horizontal", "vertical", "splits"])
    ap.add_argument("--policy", default="mpc", choices=["mpc", "mlp"])
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--segment-timesteps", type=int, default=250_000)
    ap.add_argument("--n-envs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-dir", default="/root/acmpc/runs")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--eval-freq", type=int, default=25000)
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--resume", default=None, help="checkpoint 路径（不带 .zip），env 从 <path>_env.pkl 加载")
    args = ap.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = f"ac{args.policy}_{args.track}_{args.tag}"
    ckpt_dir = os.path.join(args.log_dir, "ckpt", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume:
        print(f"[train] resume from {args.resume}", flush=True)
        with open(args.resume + "_env.pkl", "rb") as f:
            env = cloudpickle.load(f)
        model = ACMPCPPO.load(args.resume, env=env, device="cuda")
        if model.num_timesteps >= args.total_timesteps:
            print(f"[train] already at {model.num_timesteps} steps, nothing to do", flush=True)
            return
    else:
        model = make_model(args.track, args.policy, args.n_envs, args.seed, args.log_dir, run_name)

    eval_cb = TrackEvalCallback(args.track, args.eval_freq, args.eval_eps)
    ckpt_cb = CheckpointCallback(save_freq=args.eval_freq, save_path=ckpt_dir, name_prefix=run_name)

    print(f"[train] {run_name}: policy={args.policy} track={args.track} n_envs={args.n_envs} "
          f"num_timesteps={model.num_timesteps} total={args.total_timesteps} "
          f"segment={args.segment_timesteps}", flush=True)
    t0 = time.time()
    while model.num_timesteps < args.total_timesteps:
        seg = min(args.segment_timesteps, args.total_timesteps - model.num_timesteps)
        model.learn(total_timesteps=seg, callback=[eval_cb, ckpt_cb], tb_log_name=run_name,
                    reset_num_timesteps=False)
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_{model.num_timesteps}")
        save_ckpt(model, ckpt_path)
        with open(os.path.join(ckpt_dir, "LATEST"), "w") as f:
            f.write(ckpt_path)
        dt = time.time() - t0
        print(f"[train] segment done: {model.num_timesteps}/{args.total_timesteps} in {dt:.0f}s "
              f"({model.num_timesteps / dt:.0f} steps/s), ckpt={ckpt_path}", flush=True)
        if model.num_timesteps < args.total_timesteps:
            print(f"[train] exiting for restart (code {SEGMENT_EXIT_CODE})", flush=True)
            sys.exit(SEGMENT_EXIT_CODE)

    final_path = os.path.join(ckpt_dir, f"{run_name}_final")
    save_ckpt(model, final_path)
    print(f"[train] {run_name} done: final model saved to {final_path}", flush=True)


if __name__ == "__main__":
    main()
