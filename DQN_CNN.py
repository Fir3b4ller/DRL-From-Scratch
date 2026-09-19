import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_utils import ReplayBuffer, linear_schedule


def make_env(env_id: str):
    def thunk() -> gym.Env:
        env = gym.make(env_id)
        env = gym.wrappers.AtariPreprocessing(
            env, noop_max=10, frame_skip=4, screen_size=84,
            grayscale_obs=True, terminal_on_life_loss=True,
        )
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D3QN-CNN")
    parser.add_argument("--exp_name", type=str, default="D3QN-CNN")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=4000000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=500000)
    parser.add_argument("--learning_starts", type=int, default=50000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_qnet_update_freq", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=1.0)
    return parser.parse_args()


class QNetwork(nn.Module):
    """Dueling"""
    def __init__(self, obs_shape, action_dim: int):
        super().__init__()
        self.in_channels, self.h, self.w = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out = 3136
        self.fc_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
        )
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc_feature(self.conv(x / 255.0))
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class D3QNAgent:
    def __init__(self, obs_shape, action_dim: int, args: argparse.Namespace):
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNetwork(obs_shape, action_dim).to(self.device)
        self.target_qnet = QNetwork(obs_shape, action_dim).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=args.lr)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        """epsilon-greedy"""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        greedy = self.qnet(obs_t).argmax(dim=1).cpu().numpy()
        explore = np.random.rand(len(greedy)) < epsilon
        random_actions = np.random.randint(0, self.action_dim, size=len(greedy))
        return np.where(explore, random_actions, greedy).astype(np.int64)

    def update(self, batch) -> tuple[float, float]:
        s, a, r, s_, done = [t.to(self.device) for t in batch]
        a = a.unsqueeze(1)

        q = self.qnet(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            # Double DQN
            best_actions = self.qnet(s_).argmax(dim=1, keepdim=True)
            target_q = r + self.gamma * (1.0 - done) * self.target_qnet(s_).gather(1, best_actions).squeeze(1)
        loss = F.mse_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), float(q.mean().item())

    @torch.no_grad()
    def sync_target(self, tau: float) -> None:
        for p, tp in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def train(args: argparse.Namespace) -> None:
    envs = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "DQN only supports discrete action spaces"
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    obs_shape = envs.single_observation_space.shape
    action_dim = int(envs.single_action_space.n)

    buffer = ReplayBuffer(args.buffer_size)
    agent = D3QNAgent(obs_shape, action_dim, args)
    epsilon_schedule = linear_schedule(args.epsilon_start, args.epsilon_end,
                                       args.epsilon_decay_steps)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = envs.reset(seed=args.seed)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)

    global_step = 0
    updates_done = 0
    target_syncs_done = 0
    td_loss, mean_q = 0.0, 0.0
    last_step, last_time = 0, time.time()
    last_log_step = 0

    while global_step < args.total_timesteps:
        epsilon = epsilon_schedule(global_step)
        actions = agent.select_actions(obs, epsilon)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Vector envs auto-reset terminated sub-envs; true obs in `final_observation`
        real_next_obs = next_obs.copy()
        for i, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[i] = infos["final_observation"][i]

        dones = np.logical_or(terminations, truncations)
        for i in range(args.num_envs):
            buffer.add((obs[i], actions[i], rewards[i], real_next_obs[i], terminations[i]))

        obs = next_obs
        episode_returns += rewards
        episode_lengths += 1
        global_step += args.num_envs

        # log episode return and episode length
        for i in np.flatnonzero(dones):
            writer.add_scalar("charts/return", episode_returns[i], global_step)
            writer.add_scalar("charts/length", episode_lengths[i], global_step)
            print(f"global_step={global_step}, episodic_return={episode_returns[i]}")
            episode_returns[i] = 0.0
            episode_lengths[i] = 0

        # optimize the model
        if global_step >= args.learning_starts:
            n_due = (global_step - args.learning_starts) // args.train_freq
            for _ in range(n_due - updates_done):
                td_loss, mean_q = agent.update(buffer.sample(args.batch_size))
                updates_done += 1

            n_sync_due = (global_step - args.learning_starts) // args.target_qnet_update_freq
            for _ in range(n_sync_due - target_syncs_done):
                agent.sync_target(args.tau)
                target_syncs_done += 1

        if global_step - last_log_step >= 100:
            delta_steps = global_step - last_step
            elapsed = time.time() - last_time
            sps = delta_steps / elapsed
            writer.add_scalar("charts/sps", sps, global_step)
            if updates_done > 0:
                writer.add_scalar("loss/td_loss", td_loss, global_step)
                writer.add_scalar("loss/q_value", mean_q, global_step)
            print(f"SPS={int(sps)}")
            last_step, last_time = global_step, time.time()
            last_log_step = global_step

    writer.close()
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
