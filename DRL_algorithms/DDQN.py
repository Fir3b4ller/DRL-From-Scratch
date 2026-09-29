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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDQN")
    parser.add_argument("--exp_name", type=str, default="DDQN")
    parser.add_argument("--env", type=str, default="CartPole-v1") # CartPole-v1, LunarLander-v2, Acrobot-v1, MountainCar-v0
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay_steps", type=int, default=200000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_qnet_update_freq", type=int, default=500)
    parser.add_argument("--tau", type=float, default=1)
    return parser.parse_args()


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape).prod())
        action_dim = int(env.action_space.n)
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DDQNAgent:
    def __init__(self, env, args: argparse.Namespace):
        self.action_dim = int(env.action_space.n)
        self.gamma = args.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNetwork(env).to(self.device)
        self.target_qnet = QNetwork(env).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=args.lr)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """epsilon-greedy"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.qnet(obs_t)
        return int(q.argmax(dim=1).cpu().item())

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


def train(args: argparse.Namespace) -> None:
    env = gym.make(args.env)
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(env.action_space, gym.spaces.Discrete), "DQN only supports discrete action spaces"
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    buffer = ReplayBuffer(args.buffer_size)
    agent = DDQNAgent(env, args)
    epsilon_schedule = linear_schedule(args.epsilon_start, args.epsilon_end,
                                       args.epsilon_decay_steps)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = env.reset(seed=args.seed)
    episode_return, episode_length = 0.0, 0
    global_step = 0
    last_step, last_time = 0, time.time()

    while global_step < args.total_timesteps:
        epsilon = epsilon_schedule(global_step)
        action = agent.select_action(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add((obs, action, reward, next_obs, terminated))
        obs = next_obs
        episode_return += reward
        episode_length += 1
        global_step += 1

        # log episode return and episode length
        if done:
            writer.add_scalar("charts/return", episode_return, global_step)
            writer.add_scalar("charts/length", episode_length, global_step)
            print(f"global_step={global_step}, episodic_return={episode_return}")
            obs, _ = env.reset()
            episode_return, episode_length = 0.0, 0

        # optimize the model
        if global_step >= args.learning_starts:
            if global_step % args.train_freq == 0:
                td_loss, mean_q = agent.update(buffer.sample(args.batch_size))
                if global_step % 2000 == 0:
                    writer.add_scalar("loss/td_loss", td_loss, global_step)
                    writer.add_scalar("loss/q_value", mean_q, global_step)
                    delta_steps = global_step - last_step
                    elapsed = time.time() - last_time
                    writer.add_scalar("charts/sps", delta_steps / elapsed, global_step)
                    print("SPS:", int(delta_steps / elapsed))
                    last_step, last_time = global_step, time.time()

            # update target network
            if global_step % args.target_qnet_update_freq == 0:
                with torch.no_grad():
                    for p, tp in zip(agent.qnet.parameters(), agent.target_qnet.parameters()):
                        tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)