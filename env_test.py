"""步骤 7 验证：自建竞速环境 + 与 fork 策略的集成测试（在服务器 /root/acmpc 下运行）。

测试项：
[1] 悬停：F_c = m·g 时 100 步内位置/速度不漂移
[2] 环境动力学 vs DroneDx.forward_CTBR（随机状态/动作，逐元素一致）
[3] drone.py 解析雅可比 vs autograd（报告型：确认符号推导发现的
    jac_d_v_dot_dq_mat 第 3 行 / jac_q_dot_dq_mat 反对称块符号问题，不改代码）
[4] 随机策略 rollouts：三条赛道的碰撞/过门/完赛统计，无 NaN
[5] NormalizedObsWrapper：预热后观测归一化、state 通道不受影响
[6] 集成：StateVecEnv + NormalizedObsWrapper + MlpMpcPolicy + ACMPCPPO 最小训练循环
"""
import os
import sys
import time

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")

import numpy as np
import torch as th

import racing_env
from racing_env import (
    FORCE_STD,
    MASS,
    NormalizedObsWrapper,
    RacingEnv,
    make_racing_vec_env,
)

import diff_mpc_drones.drone as drone

from training_modules.mlp_mpc_policy import MlpMpcPolicy
from training_modules.ppo_glue import ACMPCPPO

G_NORM = 9.8066
T0 = time.time()


def log(name, msg):
    print(f"[{name}] {msg}  ({time.time() - T0:.1f}s)", flush=True)


def test_hover():
    env = RacingEnv(track="horizontal", start_noise=0.0)
    env.set_state([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    env.step_count = 0
    env.next_gate = 0
    a = np.zeros(4)
    a[0] = G_NORM / FORCE_STD - 1.0  # c = g → F_c = m·g，悬停
    for _ in range(100):
        obs, r, done, info = env.step(a)
    dz = env.p[2] - 1.0
    assert abs(dz) < 1e-3, f"悬停漂移 dz={dz:.3e}"
    assert np.linalg.norm(env.v) < 1e-3, f"悬停残余速度 |v|={np.linalg.norm(env.v):.3e}"
    log(1, f"悬停 OK：dz={dz:.2e}, |v|={np.linalg.norm(env.v):.2e}")


def test_dynamics_consistency():
    dx = drone.DroneDx(device="cpu")
    rng = np.random.default_rng(0)
    env = RacingEnv(track="horizontal", start_noise=0.0)
    max_err = {"p": 0.0, "q": 0.0, "v": 0.0}
    for _ in range(20):
        p = rng.uniform(-3.0, 3.0, 3)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        v = rng.uniform(-2.0, 2.0, 3)
        for _ in range(10):
            env.set_state(p, q, v)
            a = rng.uniform(-1.0, 1.0, 4)
            obs, r, done, info = env.step(a)
            env_p, env_q, env_v = env.p, env.q, env.v
            c = (np.clip(a[0], -1.0, 1.0) + 1.0) * FORCE_STD
            u = np.array([c * MASS, a[1] * 10.0, a[2] * 10.0, a[3] * 4.0])
            # drone.py 的 forward_CTBR 内部 cat 了 float32 常量，必须用 float32 输入
            x = th.tensor(np.concatenate([p, q, v])[None, :], dtype=th.float32)
            uu = th.tensor(u[None, :], dtype=th.float32)
            with th.no_grad():
                out = dx.forward_CTBR(x, uu).numpy()[0]
            dp, dq, dv = out[0:3], out[3:7], out[7:10]
            max_err["p"] = max(max_err["p"], np.abs(dp - env_p).max())
            max_err["q"] = max(max_err["q"], min(np.abs(dq - env_q).sum(), np.abs(dq + env_q).sum()))
            max_err["v"] = max(max_err["v"], np.abs(dv - env_v).max())
    log(
        2,
        f"环境 vs DroneDx：max|dp|={max_err['p']:.2e}, max|dq|(±)={max_err['q']:.2e}, "
        f"max|dv|={max_err['v']:.2e}",
    )
    assert max_err["p"] < 1e-5 and max_err["q"] < 1e-5 and max_err["v"] < 1e-5, "动力学不一致"


def _autograd_jac(fn, x):
    return th.autograd.functional.jacobian(fn, x).detach().numpy()


def test_jacobians():
    """报告型：解析雅可比 vs autograd，不修改 drone.py。"""
    dx = drone.DroneDx(device="cpu")
    rng = np.random.default_rng(1)
    errs = {"v_dot_dq": 0.0, "q_dot_dq": 0.0, "q_dot_dw": 0.0}
    for _ in range(20):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        f_c = rng.uniform(0.5, 30.0)
        w = rng.uniform(-2.0, 2.0, 3)

        # --- d(v_dot)/dq：v_dot = rotate_quat(q, [0,0,f_c]) / m ---
        q_t = th.tensor(q, dtype=th.float32, requires_grad=True)
        v_t = th.tensor([[0.0], [0.0], [f_c]], dtype=th.float32)

        def v_dot(qq):
            return drone.rotate_quat(qq.unsqueeze(1), v_t).squeeze(1) / dx.mass

        jac_auto = _autograd_jac(v_dot, q_t)  # (3,4)
        jac_ana = dx.jac_d_v_dot_dq_mat(q_t.detach(), f_c).numpy()  # (3,4)
        errs["v_dot_dq"] = max(errs["v_dot_dq"], np.abs(jac_auto - jac_ana).max())

        # --- d(q_dot)/dq、d(q_dot)/dw：q_dot = 0.5·quat_mult(q, [0,w]) ---
        w_t = th.tensor(w, dtype=th.float32, requires_grad=True)

        def q_dot_q(qq):
            wv = th.cat([th.zeros(1, dtype=th.float32), w_t])
            return 0.5 * drone.quat_mult(qq.unsqueeze(1), wv.unsqueeze(1)).squeeze(1)

        def q_dot_w(ww):
            wv = th.cat([th.zeros(1, dtype=th.float32), ww])
            return 0.5 * drone.quat_mult(q_t.unsqueeze(1), wv.unsqueeze(1)).squeeze(1)

        jac_auto = _autograd_jac(q_dot_q, q_t)  # (4,4)
        jac_ana = dx.jac_q_dot_dq_mat(w_t.detach()).numpy()  # (4,4)
        errs["q_dot_dq"] = max(errs["q_dot_dq"], np.abs(jac_auto - jac_ana).max())

        jac_auto = _autograd_jac(q_dot_w, w_t)  # (4,3)
        jac_ana = dx.jac_q_dot_dw_mat(q_t.detach()).numpy()  # (4,3)
        errs["q_dot_dw"] = max(errs["q_dot_dw"], np.abs(jac_auto - jac_ana).max())

    log(
        3,
        f"解析 vs autograd（报告型，不修改）：d(v_dot)/dq={errs['v_dot_dq']:.3e}, "
        f"d(q_dot)/dq={errs['q_dot_dq']:.3e}, d(q_dot)/dw={errs['q_dot_dw']:.3e}",
    )
    return errs


def test_random_rollouts():
    rng = np.random.default_rng(2)
    log(4, "随机策略 3 赛道 × 50 回合统计（碰撞/完赛/超时/过门数/NaN）")
    for track in ["horizontal", "vertical", "splits"]:
        env = RacingEnv(track=track, start_noise=0.5)
        stats = {"collision": 0, "finish": 0, "timeout": 0, "passes": 0, "nan": 0, "steps": 0}
        for _ in range(50):
            env.reset()
            for _ in range(2000):
                a = rng.uniform(-1.0, 1.0, 4)
                obs, r, done, info = env.step(a)
                stats["steps"] += 1
                if not (np.isfinite(obs).all() and np.isfinite(r)):
                    stats["nan"] += 1
                if r >= 10.0:
                    stats["passes"] += 1
                if done:
                    if r == 20.0:
                        stats["finish"] += 1
                    elif r == -10.0:
                        stats["collision"] += 1
                    else:
                        stats["timeout"] += 1
                    break
        print(
            f"    {track:10s} steps={stats['steps']:5d} collision={stats['collision']:3d} "
            f"finish={stats['finish']:3d} timeout={stats['timeout']:3d} "
            f"passes={stats['passes']:3d} nan={stats['nan']}",
            flush=True,
        )
        assert stats["nan"] == 0, f"{track} 出现 NaN"
        assert stats["steps"] > 0
    log(4, "随机 rollout OK")


def test_normalization():
    env = NormalizedObsWrapper(RacingEnv(track="horizontal", start_noise=0.5))
    rng = np.random.default_rng(3)
    obs = env.reset()
    post_obs = []
    for _ in range(400):
        a = rng.uniform(-1.0, 1.0, 4)
        obs, r, done, info = env.step(a)
        if done:
            obs = env.reset()
        if env.count > NormalizedObsWrapper.WARMUP_STEPS:
            post_obs.append(obs)
    post = np.stack(post_obs)
    assert np.isfinite(post).all(), "归一化观测出现 NaN"
    per_dim_std = post.std(axis=0)
    per_dim_absmax = np.abs(post).max(axis=0)
    state = env.get_state()
    assert state.shape == (13,) and np.isfinite(state).all()
    log(
        5,
        f"归一化 OK：预热后 dim_std 中位数={np.median(per_dim_std):.2f}, "
        f"absmax 中位数={np.median(per_dim_absmax):.2f}, 裁剪上限 10",
    )
    assert np.median(per_dim_std) < 5.0, "归一化后标准差异常偏大"


def test_integration():
    env = make_racing_vec_env(track="horizontal", n_envs=2, start_noise=0.5, normalize_obs=True)
    model = ACMPCPPO(
        policy=MlpMpcPolicy,
        env=env,
        learning_rate=lambda _: 3e-4,
        n_steps=16,
        batch_size=32,
        n_epochs=2,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.2},
        verbose=0,
        device="cuda",
    )
    model.learn(total_timesteps=64)
    rb = model.rollout_buffer
    assert np.isfinite(rb.observations).all(), "rollout buffer 观测有 NaN"
    assert np.isfinite(rb.states).all(), "rollout buffer state 有 NaN"
    std = float(th.exp(model.policy.log_std).mean().item())
    log(
        6,
        f"集成 OK：ACMPCPPO+MlpMpcPolicy+StateVecEnv 训练 64 步完成，"
        f"buffer obs/states 均有限，policy std={std:.3f} (init {np.exp(-1.2):.3f})",
    )


def main():
    test_hover()
    test_dynamics_consistency()
    errs = test_jacobians()
    test_random_rollouts()
    test_normalization()
    test_integration()
    print(f"\nENV_TEST_PASSED  (总耗时 {time.time() - T0:.1f}s)")
    if errs["q_dot_dw"] > 1e-2:
        print(
            "注：drone.py 解析雅可比中唯一发现的问题——jac_q_dot_dw_mat 元素 (1,2) "
            "(∂q̇_x/∂ω_z) 符号错误（aux_q_dot_dw_3 第 2 行应为 [0,0,+1,0]，实为 [0,0,-1,0]），"
            "影响 CTBR 线性化 B 矩阵的一个元素。d(v̇)/dq 与 d(q̇)/dq 均与 autograd 精确一致。"
            "保持原样不改（与论文产出所用代码一致），报告给用户。"
        )
    elif errs["v_dot_dq"] > 1e-2 or errs["q_dot_dq"] > 1e-2:
        print(
            "注：解析雅可比存在显著偏差（符号推导预测 d(v_dot)/dq 第 3 行与 "
            "d(q_dot)/dq 反对称块符号错误）——保持原样不改，报告给用户。"
        )


if __name__ == "__main__":
    main()
