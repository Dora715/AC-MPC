import os, sys
os.environ.setdefault('ACMPC_T', '5')
import numpy as np
import torch as th
from gym import spaces
sys.path.insert(0, '/root/acmpc/acmpc_public')
from training_modules.mlp_mpc_policy import MlpMpcPolicy

obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
act_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

policy = MlpMpcPolicy(obs_space, act_space, lr_schedule=lambda _: 3e-4).to('cuda')

n = 4
obs = np.zeros((n, 36), dtype=np.float32)
obs[:, 3] = 1.0                                        # quat w = 1（姿态水平）
obs[:, 9:36] = np.random.uniform(-1, 1, (n, 27)) * 0.1 # 门角点观测随机
obs_t = th.from_numpy(obs).to('cuda')

policy.eval()
with th.no_grad():
    actions, values, log_probs = policy(obs_t, obs_t, deterministic=True)

print('actions   :', actions.shape, '| 样本0:', [round(float(v), 4) for v in actions[0]])
print('values    :', values.shape, '| 样本0:', round(float(values[0]), 4))
print('log_probs :', log_probs.shape)
print('actions range check: min', round(float(actions.min()), 3), 'max', round(float(actions.max()), 3))
print('FORWARD_TEST_PASSED')
