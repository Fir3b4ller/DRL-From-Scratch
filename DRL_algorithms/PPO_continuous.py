import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id: str, gamma, normalize: bool):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO-continuous")
    parser.add_argument("--exp_name", type=str, default="PPO-continuous")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    # LunarLanderContinuous-v2, Pendulum-v1, BipedalWalker-v3, Walker2d-v4, HalfCheetah-v4, Ant-v4, Swimmer-v4, Hopper-v4
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    # PPO specific
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=256)
    return parser.parse_args()


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PPOContinuousAgent:
    def __init__(self, obs_shape, action_dim: int, action_center, action_scale, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_center = torch.as_tensor(action_center, dtype=torch.float32, device=self.device)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_eps = args.clip_eps
        self.vf_clip_eps = args.vf_clip_eps
        self.update_epochs = args.update_epochs
        self.minibatch_size = args.minibatch_size
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm

        obs_dim = int(np.array(obs_shape).prod())
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr
        )

    def _log_prob(self, obs, actions):
        mu = self.actor(obs)
        dist = torch.distributions.Normal(
            mu, self.actor.log_std.exp().expand_as(mu)
        )
        # invert action
        a = (actions - self.action_center) / self.action_scale
        a = torch.clamp(a, -1 + 1e-7, 1 - 1e-7)
        z = a.atanh()
        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-7).sum(dim=-1)
        return log_prob

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        mu = self.actor(obs_t)
        dist = torch.distributions.Normal(
            mu, self.actor.log_std.exp().expand_as(mu)
        )
        a = dist.sample().tanh()
        action = self.action_center + a * self.action_scale
        return action.cpu().numpy()

    def update(self, obs, actions, rewards, dones, last_obs):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        last_obs_t = torch.as_tensor(np.asarray(last_obs), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device)
        num_steps = obs_t.shape[0]
        num_envs = obs_t.shape[1]

        # GAE advantage
        with torch.no_grad():
            values = self.critic(obs_t)
            next_value = self.critic(last_obs_t)
            advantage = torch.zeros_like(reward)
            gae = torch.zeros_like(next_value)
            for t in reversed(range(num_steps)):
                v_next = next_value if t == num_steps - 1 else values[t + 1]
                delta = reward[t] + self.gamma * v_next * (1.0 - done[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done[t]) * gae
                advantage[t] = gae
            returns = (advantage + values).detach()
            advantage = ((advantage - advantage.mean()) / (advantage.std() + 1e-8)).detach()

        # old policy for ratio computation
        with torch.no_grad():
            old_log_prob = self._log_prob(obs_t, action)

        b_obs = obs_t.reshape(-1, obs_t.shape[-1])
        b_actions = action.reshape(-1, action.shape[-1])
        b_log_prob = old_log_prob.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantage = advantage.reshape(-1)
        b_values = values.reshape(-1)

        batch_size = num_steps * num_envs
        indices = torch.randperm(batch_size, device=self.device)
        clipfracs = []

        for _ in range(self.update_epochs):
            for start in range(0, batch_size, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                mb_obs = b_obs[idx]
                mb_actions = b_actions[idx]
                mb_old_log_prob = b_log_prob[idx]
                mb_returns = b_returns[idx]
                mb_advantage = b_advantage[idx]
                mb_values = b_values[idx]

                log_prob = self._log_prob(mb_obs, mb_actions)
                entropy = -log_prob.mean()

                ratio = torch.exp(log_prob - mb_old_log_prob)
                with torch.no_grad():
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                clipfracs.append(torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item())

                # policy loss
                surr1 = ratio * mb_advantage
                surr2 = clipped_ratio * mb_advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (clipped)
                new_values = self.critic(mb_obs)
                value_pred_clipped = mb_values + torch.clamp(
                    new_values - mb_values, -self.vf_clip_eps, self.vf_clip_eps
                )
                value_losses = (new_values - mb_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

        return (
            policy_loss.item(),
            value_loss.item(),
            entropy.item(),
            np.mean(clipfracs),
        )


def train(args: argparse.Namespace):
    envs = gym.vector.AsyncVectorEnv([make_env(args.env, args.gamma, args.normalize) for _ in range(args.num_envs)])
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only supports continuous action spaces"

    obs_shape = envs.single_observation_space.shape
    action_dim = int(envs.single_action_space.shape[0])

    low = np.asarray(envs.single_action_space.low, dtype=np.float32)
    high = np.asarray(envs.single_action_space.high, dtype=np.float32)
    action_center = (high + low) / 2.0
    action_scale = (high - low) / 2.0

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    agent = PPOContinuousAgent(obs_shape, action_dim, action_center, action_scale, args)

    # log hyperparameters
    writer = SummaryWriter(f"runs/{run_name}")
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    start_time = time.time()
    last_log_step = 0

    while global_step < args.total_timesteps:
        rollout_obs, rollout_actions = [], []
        rollout_rewards, rollout_dones = [], []

        for _ in range(args.num_steps):
            actions = agent.select_actions(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)

            rollout_obs.append(obs.copy())
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)

            obs = next_obs
            global_step += args.num_envs

            for i in np.flatnonzero(dones):
                episode_reward = float(infos["final_info"][i]["episode"]["r"])
                episode_length = int(infos["final_info"][i]["episode"]["l"])
                writer.add_scalar("charts/return", episode_reward, global_step)
                writer.add_scalar("charts/length", episode_length, global_step)
                print(f"global_step={global_step}, episodic_return={episode_reward:.3f}")

            if global_step - last_log_step >= 2000:
                sps = global_step / (time.time() - start_time)
                writer.add_scalar("charts/sps", sps, global_step)
                print(f"SPS={int(sps)}")
                last_log_step = global_step

        # update
        if args.anneal_lr:
            frac = 1.0 - global_step / args.total_timesteps
            lr_now = args.lr * frac
            for group in agent.optimizer.param_groups:
                group["lr"] = lr_now
            writer.add_scalar("charts/learn_rate", lr_now, global_step)

        policy_loss, value_loss, entropy, clipfrac = agent.update(
            np.stack(rollout_obs),
            np.stack(rollout_actions),
            np.stack(rollout_rewards),
            np.stack(rollout_dones),
            obs,
        )
        writer.add_scalar("loss/policy_loss", policy_loss, global_step)
        writer.add_scalar("loss/value_loss", value_loss, global_step)
        writer.add_scalar("loss/entropy", entropy, global_step)
        writer.add_scalar("loss/clipfrac", clipfrac, global_step)

    writer.close()
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)