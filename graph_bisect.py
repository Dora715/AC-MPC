"""图路径二分：真实 forward_actor 输出的 requires_grad 状态。

再复制一份 forward_actor 逻辑，分别对 _Q/_p 填充与 nom_u 回填做 detach 开关，
确定图到底在哪里接通到 policy_net。
"""
import os
import sys

os.environ.setdefault("ACMPC_T", "5")
sys.path.insert(0, "/root/acmpc/acmpc_public")
sys.path.insert(0, "/root/acmpc/acmpc_public/training_modules")
sys.path.insert(0, "/root/acmpc/acmpc_public/diff_mpc_drones")

import numpy as np
import torch as th
from gym import spaces

import drone
import il_env
from training_modules.mlp_mpc_policy import MlpMpcPolicy


def build():
    policy = MlpMpcPolicy(
        spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        lambda _: 3e-4,
    )
    policy.to("cuda")
    policy.set_training_mode(False)
    return policy


def main():
    policy = build()
    n = 64
    obs = th.randn(n, 36, device="cuda")
    states = th.randn(n, 13, device="cuda")

    # A: 真实 forward_actor
    out = policy.mlp_extractor.forward_actor(obs, states)
    print(f"A real forward_actor: out.requires_grad={out.requires_grad} "
          f"grad_fn={type(out.grad_fn).__name__ if out.grad_fn else None}")

    # B: 复制版，带开关
    ext = policy.mlp_extractor

    def copy_forward(features, states, detach_q, detach_nomu):
        states = states.to(ext.device).float()
        features_in = features[:, :ext.features_in_dim]
        states = states[:, 0:10]
        sigmoid_cost_all = ext.policy_net(features_in)
        n_batch = features.shape[0]
        chunk_length = 1024
        chunks = th.split(sigmoid_cost_all, chunk_length, dim=0)
        epsilon = 0.1
        range_Q = 100000.0
        range_p = 100000.0
        range_p_t = 2 * range_Q / 2 * ext.dx.mass * 9.806
        n_tau = 14
        if ext.u_prev is None:
            ext.u_prev = th.zeros(4, n_batch).to(device=ext.device)
            ext.u_prev[0, :] = ext.dx.mass * 9.806
        nom_x = th.zeros((n_batch, ext.T, ext.dx.n_state)).to(device=ext.device)
        nom_u = th.zeros((n_batch, ext.T, ext.dx.n_ctrl)).to(device=ext.device)
        idx_start = 0
        for idx, sigmoid_cost in enumerate(chunks):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            x_Q = sigmoid_cost[:, :14*ext.T].to(device=ext.device)
            x_p = sigmoid_cost[:, 14*ext.T:].to(device=ext.device)
            q_p = x_Q[:, :3*ext.T] * range_Q + epsilon
            q_q = x_Q[:, 3*ext.T:7*ext.T] * range_Q + epsilon
            q_v = x_Q[:, 7*ext.T:10*ext.T] * range_Q + epsilon
            q_w = x_Q[:, 10*ext.T:13*ext.T] * range_Q + epsilon
            q_t = x_Q[:, 13*ext.T:14*ext.T] * range_Q + epsilon
            p_p = (x_p[:, :3*ext.T] - 0.5) * range_p
            p_q = (x_p[:, 3*ext.T:7*ext.T] - 0.5) * range_p
            p_v = (x_p[:, 7*ext.T:10*ext.T] - 0.5) * range_p
            p_w = (x_p[:, 10*ext.T:13*ext.T] - 0.5) * range_p
            p_t = x_p[:, 13*ext.T:14*ext.T] * range_p_t + epsilon
            u_prev_chunk = ext.u_prev[:, idx_start:idx_end]
            _Q = th.zeros(ext.T, n_chunk, n_tau, n_tau, device=ext.device)
            _p = th.zeros(ext.T, n_chunk, n_tau, device=ext.device)
            states_chunk = states[idx_start:idx_end, :]
            for i in range(ext.T):
                Q_diag_embed_i = th.diag_embed(th.cat([
                    q_p[:, i*3:i*3+3], q_q[:, i*4:i*4+4], q_v[:, i*3:i*3+3],
                    q_t[:, i].unsqueeze(1), q_w[:, i*3:i*3+3]], dim=1))
                p_i = th.cat([p_p[:, i*3:i*3+3], p_q[:, i*4:i*4+4],
                              p_v[:, i*3:i*3+3], -p_t[:, i].unsqueeze(1),
                              p_w[:, i*3:i*3+3]], dim=1)
                if detach_q:
                    Q_diag_embed_i = Q_diag_embed_i.detach()
                    p_i = p_i.detach()
                _Q[i, :, :, :] = Q_diag_embed_i
                _p[i, :, :] = p_i
            nom_x_chunk, nom_u_chunk = ext.mpc_env.mpc(
                ext.dx, states_chunk, _Q, _p, u_init=u_prev_chunk,
                lqr_iter_override=1)
            if detach_nomu:
                nom_x_chunk = nom_x_chunk.detach()
                nom_u_chunk = nom_u_chunk.detach()
            nom_x[idx_start:idx_end, :, :] = nom_x_chunk.transpose(0, 1)
            nom_u[idx_start:idx_end, :, :] = nom_u_chunk.transpose(0, 1)
            idx_start = idx_end
        ext.u_prev = nom_u[:, 0, :].transpose(0, 1)
        ext.predictions = th.cat((nom_x, nom_u), dim=2).detach()
        thrust = nom_u[:, 0, 0] / ext.dx.mass
        omegas = nom_u[:, 0, 1:4]
        force_mean = (8.5 * 4 / ext.dx.mass) / 2.0
        thrust_normalized = (thrust - force_mean) / force_mean
        omega_max = th.Tensor([10.0, 10.0, 4.0]).to(device=ext.device)
        omegas_normalized = th.div(omegas, omega_max).to(device=ext.device)
        return th.cat((thrust_normalized.unsqueeze(1), omegas_normalized), dim=1).to(ext.device)

    for detach_q, detach_nomu in [(False, False), (True, False), (False, True), (True, True)]:
        out = copy_forward(obs, states, detach_q, detach_nomu)
        print(f"B copy detach_q={detach_q} detach_nomu={detach_nomu}: "
              f"requires_grad={out.requires_grad} "
              f"grad_fn={type(out.grad_fn).__name__ if out.grad_fn else None}")

    print("GRAPH_BISECT_DONE", flush=True)


if __name__ == "__main__":
    main()
