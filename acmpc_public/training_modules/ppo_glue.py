"""PPO 子类胶水：给 SB3-acmpc fork 的四处内部不一致打补丁。

SB3-acmpc fork 把 states 链路改成了这样，但算法类的调用点没有跟着改：

  1. 策略接口: forward(obs, states, deterministic) / evaluate_actions(obs, actions, states)
     （states 的前 10 维是 [p, q, v]，MlpMpcPolicy 用它作为 MPC 的初始状态）
     但 collect_rollouts 调 self.policy(obs_tensor)，train 调
     evaluate_actions(obs, actions)，都会抛 TypeError。
  2. RolloutBuffer 构造需要 state_space 参数（13 维，硬编码 state_dim=13），
     但 OnPolicyAlgorithm._setup_model 没传。
  3. RolloutBuffer.add(obs, action, state, reward, episode_start, value, log_prob)
     中间多了 state 参数，但 collect_rollouts 的 add 调用还是旧签名。
  4. MlpMpcPolicy.mlp_extractor.u_prev（MPC 热启动，形状 = 上次前向的 batch）
     在 rollout(batch=n_envs) 与 train(batch=batch_size) 之间切换时可能越界；
     且 il_env.mpc 每次调用都重建 u_init，传入的热启动实为死代码，
     故 batch 变大时直接重置（见 _clear_stale_u_prev）。
  5. _setup_learn 里 self._last_obs, self._last_state = self.env.reset()，
     假设 reset() 返回 (obs, state) 元组（其未发布的环境包装器才这样）；
     标准 VecEnv 只返回 obs，会被按第 0 维解包成单环境的 1 维观测。

state 来源：env 提供 get_state() 时用其专用 13 维通道
（StateVecEnv 返回 [p,q,v,下一门中心相对位置]）；否则回退为 obs 的前 13 维
（前 10 维是 [p,q,v]，供 glue_test 的哑环境使用）。这里子类化 PPO，
把相关方法原样复制并补上 state 传参，不改动 fork 的原始代码。
"""
import time
from collections import deque
from typing import Tuple

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common import utils
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

STATE_DIM = 13  # fork 的 RolloutBuffer 硬编码 state_dim=13


class ACMPCPPO(PPO):
    @staticmethod
    def _clear_stale_u_prev(policy, n_cols: int) -> None:
        """MlpMpcPolicy 的 mlp_extractor.u_prev 保存上一次前向的 MPC 热启动序列，
        形状 = 上一次前向的 batch 大小；下一次前向的 batch 更大时，切片
        self.u_prev[:, idx_start:idx_end] 会越界。而 il_env.mpc 每次调用都重建
        u_init（悬停初值），传入的热启动实际被忽略（死代码），所以 batch 列数
        不够时直接重置为 None 让其惰性重建，不影响任何行为。
        """
        u_prev = getattr(policy.mlp_extractor, "u_prev", None)
        if u_prev is not None and u_prev.shape[1] < n_cols:
            policy.mlp_extractor.u_prev = None

    def _setup_model(self) -> None:
        """复制自 OnPolicyAlgorithm._setup_model + PPO._setup_model。

        唯一改动：构造 RolloutBuffer 时补传 fork 要求的 state_space 参数。
        """
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32),
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # 以下复制自 PPO._setup_model：初始化 clip range 调度
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """复制自 BaseAlgorithm._setup_learn。

        唯一改动：fork 里 self._last_obs, self._last_state = self.env.reset()
        假设 reset() 返回 (obs, state) 元组；标准 VecEnv 只返回 obs (n_envs, dim)，
        按第 0 维解包会把 _last_obs 拆成单环境的 1 维观测。这里改成：
        返回元组时照旧解包；否则 obs 全量赋给 _last_obs，
        _last_state = obs 的前 13 维。
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            # PATCH: 兼容两种 reset() 返回值（见 docstring）
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                self._last_obs, self._last_state = reset_result
            else:
                self._last_obs = reset_result
                self._last_state = reset_result[:, :STATE_DIM]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """复制自 OnPolicyAlgorithm.collect_rollouts。

        三处改动：
        1. self.policy(obs_tensor) -> self.policy(obs_tensor, state_tensor)，
           补传 fork 策略接口要求的 states（state 来自 _last_state 通道）；
        2. rollout_buffer.add(...) 按 fork 的新签名补传 state（_last_state）；
        3. batch 变大前重置过期的 u_prev 热启动（见 _clear_stale_u_prev）。
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        # PATCH: rollout 的 batch 是 n_envs，若上次训练留下的 u_prev 列数不够则重置
        self._clear_stale_u_prev(self.policy, env.num_envs)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # PATCH: 补传 states（fork 的策略接口要求 forward(obs, states)）。
                # state 走专用通道（_last_state）；StateVecEnv 每步后由
                # get_state() 同步，此处直接转 tensor。
                state_tensor = th.as_tensor(self._last_state, dtype=th.float32).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor, state_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # PATCH: 从 env 的专用通道取 step 后的 state；无该通道时回退
            # 到 obs 前 13 维（哑环境 / 标准 VecEnv）。
            if hasattr(env, "get_state"):
                new_state = env.get_state()
            else:
                new_state = new_obs[:, :STATE_DIM]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # PATCH: fork 的 add 签名是 (obs, action, state, reward, episode_start, value, log_prob)
            rollout_buffer.add(
                self._last_obs, actions, self._last_state, rewards, self._last_episode_starts, values, log_probs
            )
            self._last_obs = new_obs
            self._last_state = new_state
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """复制自 PPO.train。

        唯一改动：evaluate_actions(obs, actions) ->
        evaluate_actions(obs, actions, rollout_data.states)，
        states 是 rollout 时存入缓冲区的 13 维状态。
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # PATCH: train 的 batch 是 batch_size，若上次 rollout 留下的 u_prev 列数不够则重置
        self._clear_stale_u_prev(self.policy, self.batch_size)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # PATCH: 补传 states（fork 的策略接口要求 evaluate_actions(obs, actions, states)），
                # 来自 RolloutBuffer 在 rollout 时存储的 13 维 state
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, rollout_data.states
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
