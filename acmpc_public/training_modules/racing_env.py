"""自建四旋翼穿门竞速环境（AC-MPC 方法级复现，步骤 7）。

设计依据（论文 IV-A 节实验设置）：
- 观测 obs（36 维）= [v(3), R(9, 行主序), 2 个未来门 × 4 角 × 相对位置(3)]
  与论文一致（o_quad = [v, R]，otrack = 每门 4 角相对位置，G=2）；不含位置，
  位置由 MPC 的 state 通道提供。
- 状态 state（13 维）= [p(3), q(4), v(3), 下一门中心相对位置(3)]，由
  StateVecEnv 单独通道返回（对应 fork 里 env.reset() 返回 (obs, state)、
  RolloutBuffer 硬编码 state_dim=13 的设计）。MlpMpcPolicy 取 states[:, :10]
  作为 MPC 初始状态。
- 动作（4 维，∈[−1,1]）= 质量归一化推力 c 与机体角速率 ω 的归一化值，
  反归一化常数与 mlp_mpc_policy.forward_actor 完全一致：
  c = (a0+1)·(8.5·4/mass)/2 [m/s²]，F_c = c·mass [N]；ω = a[1:]·[10,10,4] [rad/s]。
- 奖励 eq.(10)：碰撞 −10 / 过门 +10 / 完赛 +10 / 其余
  ‖g_k − p_{k−1}‖ − ‖g_k − p_k‖ − 0.01‖ω_k‖（g_k 为当前目标门中心，b=0.01）。
  过最后一门时过门与完赛同时触发，合计 +20。
- 动力学：与 diff_mpc_drones/drone.py 的 DroneDx.forward_CTBR 逐式一致
  （CTBR：ṗ=v，q̇=q⊗exp(ω·dt/2)，v̇=R(q)·e3·(F_c/m)+g，dt=0.02），
  保证环境动力学 = MPC 内部模型。
- 观测归一化：NormalizedObsWrapper 对全部 36 维做增量运行统计归一化
  （前 100 步预热输出原始观测，随后裁剪到 ±10）。
  论文为"每训练迭代用该迭代数据算均值/标准差"，此处用逐步运行统计近似
  （因果、可在线），state 通道保持原始物理量不受归一化影响。
"""
import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.vec_env import DummyVecEnv

# ---- 与 drone.py 一致的常数 ----
G = np.array([0.0, 0.0, -9.8066])
MASS = 0.752
DT = 0.02
NORMALIZATION_MAX = 8.5  # 单旋翼最大推力 [N]
FORCE_MEAN = FORCE_STD = (NORMALIZATION_MAX * 4.0 / MASS) / 2.0  # ≈22.61 m/s²
OMEGA_MAX = np.array([10.0, 10.0, 4.0])
THRUST_MAX = NORMALIZATION_MAX * 4.0  # 34 N

GATE_SIDE = 1.5
HALF_SIDE = GATE_SIDE / 2.0
PASS_TOL = 0.15  # 门框判定容差（离散时间穿越检测的补偿）
MAX_EPISODE_STEPS = 500  # 10 s
BOUND_RADIUS = 40.0  # 飞出安全边界视为碰撞

STATE_DIM = 13
OBS_DIM = 36
RATE_PENALTY = 0.01

TRACKS = {
    # 水平平飞（论文的 easy track）：同高度直线门，带轻微高低起伏
    "horizontal": [
        dict(center=(2.0, 0.0, 1.5), yaw=0.0),
        dict(center=(4.5, 0.0, 1.0), yaw=0.0),
        dict(center=(7.0, 0.0, 1.5), yaw=0.0),
        dict(center=(9.5, 0.0, 1.0), yaw=0.0),
        dict(center=(12.0, 0.0, 1.5), yaw=0.0),
    ],
    # 垂直爬升（论文的 vertical track）：陡峭上升，考验推力饱和
    "vertical": [
        dict(center=(1.0, 0.0, 2.0), yaw=0.0),
        dict(center=(1.0, 0.0, 4.5), yaw=0.0),
        dict(center=(1.0, 0.0, 7.0), yaw=0.0),
        dict(center=(1.0, 0.0, 9.5), yaw=0.0),
    ],
    # SplitS（论文的 hard track）：两门上下重叠，上穿高门后需半筋斗俯冲反穿低门
    "splits": [
        dict(center=(2.0, 0.0, 1.5), yaw=0.0),
        dict(center=(4.0, 0.0, 2.5), yaw=0.0),
        dict(center=(6.0, 0.0, 3.5), yaw=0.0),
        dict(center=(6.0, 0.0, 1.0), yaw=np.pi),
        dict(center=(3.0, 0.0, 1.0), yaw=np.pi),
    ],
}


def quat_mult(q1, q2):
    """drone.py quat_mult 的 numpy 版：返回标准 Hamilton 积 q1 ⊗ q2。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def exp_map(w, dt):
    """角速率 w（机体系）在 dt 内对应的增量四元数（与 drone.py exp_map 一致）。"""
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    w_hat = w / w_norm
    half = w_norm * dt * 0.5
    return np.array([np.cos(half), w_hat[0] * np.sin(half), w_hat[1] * np.sin(half), w_hat[2] * np.sin(half)])


def rotate(q, v):
    """R_std(q)·v（标准主动旋转矩阵，与 drone.py rotate_quat 一致）。"""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ v


class Gate:
    def __init__(self, center, yaw):
        self.center = np.asarray(center, dtype=np.float64)
        self.yaw = float(yaw)
        self.n = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])  # 通过方向（门法线）
        self.t1 = np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])  # 门平面横轴
        self.t2 = np.array([0.0, 0.0, 1.0])  # 门平面竖轴
        self.corners = np.array(
            [self.center + s1 * HALF_SIDE * self.t1 + s2 * HALF_SIDE * self.t2 for s1 in (-1, 1) for s2 in (-1, 1)]
        )


class RacingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, track="horizontal", gate_side=GATE_SIDE, start_pos=(0.0, 0.0, 1.0), start_noise=0.5):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.gate_side = gate_side
        self.half_side = gate_side / 2.0
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.start_noise = start_noise
        if track not in TRACKS:
            raise ValueError(f"Unknown track '{track}', available: {list(TRACKS)}")
        self.track_name = track
        self.gates = [Gate(**cfg) for cfg in TRACKS[track]]
        self.n_gates = len(self.gates)
        self.reset()

    def reset(self):
        # 初始位置在名义起点 ±start_noise 立方体内随机（论文：训练时 1m 立方体）
        p = self.start_pos + np.random.uniform(-self.start_noise, self.start_noise, 3)
        self.p = p
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v = np.zeros(3)
        self.next_gate = 0
        self.step_count = 0
        self.ep_reward = 0.0
        self.last_action_omega = np.zeros(3)
        return self._build_obs()

    def set_state(self, p, q, v):
        """测试用：直接设置完整状态。"""
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
        self.v = np.asarray(v, dtype=np.float64)

    def get_state(self):
        """13 维 state：喂给 MPC 的 [p,q,v] + 下一门中心相对位置。"""
        gate_center = self.gates[min(self.next_gate, self.n_gates - 1)].center
        return np.concatenate([self.p, self.q, self.v, gate_center - self.p]).astype(np.float32)

    def denormalize_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        c = (action[0] + 1.0) * FORCE_STD
        omega = action[1:] * OMEGA_MAX
        return c, omega

    def _integrate(self, u):
        """与 DroneDx.forward_CTBR 逐式一致：u = [F_c, ωx, ωy, ωz]（牛顿）。"""
        F_c, wx, wy, wz = u
        w = np.array([wx, wy, wz])
        new_p = self.p + DT * self.v
        em = exp_map(w, DT)
        new_q = quat_mult(self.q, em)
        new_q = new_q / np.linalg.norm(new_q)
        new_v = self.v + DT * (rotate(self.q, np.array([0.0, 0.0, F_c / MASS])) + G)
        self.p, self.q, self.v = new_p, new_q, new_v

    def step(self, action):
        c, omega = self.denormalize_action(action)
        p_prev = self.p.copy()
        self.last_action_omega = omega
        u = np.array([c * MASS, omega[0], omega[1], omega[2]])
        self._integrate(u)
        self.step_count += 1

        reward, done = self._events(p_prev, omega)
        self.ep_reward += reward
        info = {
            "gates_passed": self.next_gate,
            "collision": done and self.next_gate < self.n_gates and reward < 0,
        }
        if done:
            info["is_success"] = self.next_gate >= self.n_gates
            info["episode"] = {"r": float(self.ep_reward), "l": self.step_count}
        return self._build_obs(), reward, done, info

    def _events(self, p_prev, omega):
        # 撞地 / 飞出边界 → 碰撞
        if self.p[2] <= 0.0 or np.linalg.norm(self.p) > BOUND_RADIUS:
            return -10.0, True

        gate = self.gates[self.next_gate]
        d_prev = float((p_prev - gate.center) @ gate.n)
        d_new = float((self.p - gate.center) @ gate.n)

        # 穿过门平面（法线方向由负变正）
        if d_prev < 0.0 <= d_new:
            delta = self.p - gate.center
            if abs(float(delta @ gate.t1)) <= self.half_side + PASS_TOL and abs(
                float(delta @ gate.t2)
            ) <= self.half_side + PASS_TOL:
                self.next_gate += 1
                if self.next_gate >= self.n_gates:
                    return 20.0, True  # 过最后一门 +10，完赛 +10
                return 10.0, False  # 过门 +10
            return -10.0, True  # 撞门框

        # 进度奖励：朝目标门中心逼近的距离 − 角速率惩罚
        progress = np.linalg.norm(gate.center - p_prev) - np.linalg.norm(gate.center - self.p)
        progress -= RATE_PENALTY * np.linalg.norm(omega)
        if self.step_count >= MAX_EPISODE_STEPS:
            return progress, True
        return progress, False

    def _build_obs(self):
        v = self.v
        R = rotate_mat(self.q)  # (3,3)
        nxt = min(self.next_gate, self.n_gates - 1)
        nxt2 = min(self.next_gate + 1, self.n_gates - 1)
        corners1 = (self.gates[nxt].corners - self.p).reshape(-1)  # 12
        corners2 = (self.gates[nxt2].corners - self.p).reshape(-1)  # 12
        obs = np.concatenate([v, R.reshape(-1), corners1, corners2])  # 3+9+24 = 36
        return obs.astype(np.float32)


def rotate_mat(q):
    """R_std(q) 旋转矩阵（行主序）。"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class NormalizedObsWrapper(gym.Wrapper):
    """对全部 36 维观测做运行统计归一化（state 通道不受影响）。

    前 WARMUP_STEPS 步输出原始观测以积累统计量，之后输出裁剪到 ±CLIP 的
    归一化观测。论文为每训练迭代批量归一化，这里用逐样本运行统计近似。
    """

    WARMUP_STEPS = 100
    CLIP = 10.0
    EPS = 1e-3

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.count = 0.0
        self.mean = np.zeros(OBS_DIM, dtype=np.float64)
        self.m2 = np.zeros(OBS_DIM, dtype=np.float64)

    def _update(self, obs):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.m2 += delta * delta2

    def _set_stats(self, mean, count, m2):
        """评估用：直接注入训练环境的运行统计。"""
        self.mean = np.asarray(mean, dtype=np.float64).copy()
        self.count = float(count)
        self.m2 = np.asarray(m2, dtype=np.float64).copy()

    def _norm(self, obs):
        if self.count <= self.WARMUP_STEPS:
            return obs
        std = np.sqrt(self.m2 / self.count + self.EPS)
        return np.clip((obs - self.mean) / std, -self.CLIP, self.CLIP).astype(np.float32)

    def reset(self):
        obs = self.env.reset()
        self._update(obs)
        return self._norm(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update(obs)
        return self._norm(obs), reward, done, info

    def get_state(self):
        return self.env.get_state()


class StateVecEnv(DummyVecEnv):
    """在 DummyVecEnv 之上增加 13 维 state 通道。

    reset() 返回 (obs, state)（对应 fork _setup_learn 的元组解包设计）；
    step() 返回标准 4 元组；step 后可用 get_state() 取当前 state。
    """

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self._states = np.zeros((self.num_envs, STATE_DIM), dtype=np.float32)

    def reset(self):
        obs = super().reset()
        self._sync_states()
        return obs, self._states.copy()

    def step_wait(self):
        obs, rews, dones, infos = super().step_wait()
        self._sync_states()
        return obs, rews, dones, infos

    def _sync_states(self):
        for idx, env in enumerate(self.envs):
            self._states[idx] = env.get_state()

    def get_state(self):
        return self._states


def make_racing_vec_env(track="horizontal", n_envs=4, start_noise=0.5, normalize_obs=True):
    def make_env(seed=None):
        def _init():
            env = RacingEnv(track=track, start_noise=start_noise)
            if normalize_obs:
                env = NormalizedObsWrapper(env)
            return env

        return _init

    return StateVecEnv([make_env() for _ in range(n_envs)])
