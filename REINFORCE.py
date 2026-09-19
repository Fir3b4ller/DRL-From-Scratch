import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id: str):
    def thunk() -> gym.Env:
        return gym.make(env_id)
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE")
    parser.add_argument("--exp_name", type=str, default="REINFORCE")
    parser.add_argument("--env", type=str, default="CartPole-v1") # CartPole-v1, LunarLander-v2, Acrobot-v1
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    return parser.parse_args()


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReinforceAgent:
    def __init__(self, obs_shape, action_dim: int, args: argparse.Namespace):
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = int(np.array(obs_shape).prod())
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().cpu().numpy()

    def update(self, states, actions, rewards):
        """REINFORCE"""
        s = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.stack(actions), dtype=torch.long, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(r)
        g = 0.0
        for t in range(len(r) - 1, -1, -1):
            g = r[t] + self.gamma * g
            returns[t] = g

        logits = self.policy(s)
        log_probs = F.log_softmax(logits, dim=1).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train(args: argparse.Namespace) -> None:
    envs = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only supports discrete action spaces"

    obs_shape = envs.single_observation_space.shape
    action_dim = int(envs.single_action_space.n)

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    agent = ReinforceAgent(obs_shape, action_dim, args)

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
            episode_reward = sum(ep_rewards[i])
            episode_length = len(ep_obs[i])
            policy_loss = agent.update(ep_obs[i], ep_actions[i], ep_rewards[i])
            writer.add_scalar("charts/return", episode_reward, global_step)
            writer.add_scalar("charts/length", episode_length, global_step)
            writer.add_scalar("loss/policy_loss", policy_loss, global_step)
            print(f"global_step={global_step}, episodic_return={episode_reward}")
            ep_obs[i].clear()
            ep_actions[i].clear()
            ep_rewards[i].clear()

        if global_step - last_log_step >= 100:
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