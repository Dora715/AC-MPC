import torch, sys
sys.path.insert(0, '/root/acmpc/acmpc_public/diff_mpc_drones')
import drone, il_env

dx = drone.DroneDx()
env = il_env.IL_Env('drone', mpc_T=5)

n_batch = 2
xinit = torch.zeros(n_batch, dx.n_state)
xinit[:, 3] = 1.0

T = 5
n_tau = dx.n_state + dx.n_ctrl
Q = torch.zeros(T, n_batch, n_tau, n_tau)
p = torch.zeros(T, n_batch, n_tau)
for i in range(T):
    Q[i] = torch.eye(n_tau) * 0.1
Q.requires_grad_()
p.requires_grad_()

x_mpc, u_mpc = env.mpc(dx, xinit, Q, p, lqr_iter_override=5)
print('x_mpc:', tuple(x_mpc.shape), ' u_mpc:', tuple(u_mpc.shape))
print('u[0] (thrust, wx, wy, wz):', [round(float(v), 3) for v in u_mpc[0, 0]])

loss = u_mpc.sum()
loss.backward()
print('Q.grad finite:', Q.grad is not None and torch.isfinite(Q.grad).all().item())
print('p.grad finite:', p.grad is not None and torch.isfinite(p.grad).all().item())
print('SMOKE_TEST_PASSED')
