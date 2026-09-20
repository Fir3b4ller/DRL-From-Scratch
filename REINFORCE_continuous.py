import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id: str, gamma):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE-continuous")
    parser.add_argument("--exp_name", type=str, default="REINFORCE-continuous")
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2") # LunarLanderContinuous-v2
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    return parser.parse_args()


class PolicyNetwork(nn.Module):
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
        return torch.tanh(self.network(x))


class ReinforceContinuousAgent:
    def __init__(self, obs_shape, action_dim: int, action_lim: float, args: argparse.Namespace):
        self.action_lim = action_lim
        self.gamma = args.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = int(np.array(obs_shape).prod())
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        mean = self.policy(obs_t) * self.action_lim
        dist = torch.distributions.Normal(mean, self.policy.log_std.exp().expand_as(mean))
        return dist.sample().cpu().numpy()

    def update(self, states, actions, rewards):
        """REINFORCE"""
        s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(r)
        g = 0.0
        for t in range(len(r) - 1, -1, -1):
            g = r[t] + self.gamma * g
            returns[t] = g

        mean = self.policy(s) * self.action_lim
        dist = torch.distributions.Normal(mean, self.policy.log_std.exp().expand_as(mean))
        log_probs = dist.log_prob(a).sum(dim=1)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train(args: argparse.Namespace) -> None:
    envs = gym.vector.AsyncVectorEnv([make_env(args.env, args.gamma) for _ in range(args.num_envs)])
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only supports discrete action spaces"

    obs_shape = envs.single_observation_space.shape
    action_dim = int(envs.single_action_space.shape[0])
    action_lim = float(envs.single_action_space.high[0])

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    agent = ReinforceContinuousAgent(obs_shape, action_dim, action_lim, args)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    ep_obs = [[] for _ in range(args.num_envs)]
    ep_actions = [[] for _ in range(args.num_envs)]
    ep_rewards = [[] for _ in range(args.num_envs)]

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    last_step, last_time = 0, time.time()
    last_log_step = 0

    while global_step < args.total_timesteps:
        actions = agent.select_actions(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        for i in range(args.num_envs):
            ep_obs[i].append(obs[i])
            ep_actions[i].append(actions[i])
            ep_rewards[i].append(rewards[i])

        obs = next_obs
        global_step += args.num_envs

        for i in np.flatnonzero(dones):
            episode_reward = float(infos["final_info"][i]["episode"]["r"])
            episode_length = int(infos["final_info"][i]["episode"]["l"])
            policy_loss = agent.update(ep_obs[i], ep_actions[i], ep_rewards[i])
            writer.add_scalar("charts/return", episode_reward, global_step)
            writer.add_scalar("charts/length", episode_length, global_step)
            writer.add_scalar("loss/policy_loss", policy_loss, global_step)
            print(f"global_step={global_step}, episodic_return={episode_reward:.3f}")
            ep_obs[i].clear()
            ep_actions[i].clear()
            ep_rewards[i].clear()

        if global_step - last_log_step >= 2000:
            delta_steps = global_step - last_step
            elapsed = time.time() - last_time
            sps = delta_steps / elapsed
            writer.add_scalar("charts/sps", sps, global_step)
            print(f"SPS={int(sps)}")
            last_step, last_time = global_step, time.time()
            last_log_step = global_step

    writer.close()
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)