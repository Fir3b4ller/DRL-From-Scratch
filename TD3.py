import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_utils import ReplayBuffer


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


def parse_args():
    parser = argparse.ArgumentParser(description="TD3")
    parser.add_argument("--exp_name", type=str, default="TD3")
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2")
    # Pendulum-v1, LunarLanderContinuous-v2, BipedalWalker-v3, Walker2d-v4, HalfCheetah-v4, Ant-v4, Swimmer-v4, Hopper-v4
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([x, a], dim=-1)).squeeze(-1)


class TD3Agent:
    def __init__(self, obs_shape, action_dim: int, action_center, action_scale, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_center = torch.as_tensor(action_center, dtype=torch.float32, device=self.device)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        self.action_low = torch.as_tensor(action_center - action_scale, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_center + action_scale, dtype=torch.float32, device=self.device)
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

        obs_dim = int(np.array(obs_shape).prod())
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.lr)

    def _convert(self, u: torch.Tensor) -> torch.Tensor:
        """map tanh output u in [-1, 1] to the real action space"""
        return self.action_center + u * self.action_scale

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """deterministic policy (no exploration noise; handled by target smoothing)"""
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self._convert(self.actor(obs_t))
        return action.cpu().numpy().squeeze(0)

    def update(self, batch, update_actor: bool):
        s, a, r, s_, done = [t.to(self.device) for t in batch]

        # twin critic target: target policy smoothing + min(Q1', Q2')
        with torch.no_grad():
            next_action = self._convert(self.target_actor(s_))
            noise = torch.randn_like(next_action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, self.action_low, self.action_high)
            target_q = r + self.gamma * (1.0 - done) * torch.min(
                self.target_critic1(s_, next_action), self.target_critic2(s_, next_action)
            )

        # critic loss (both critics)
        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delayed actor / target updates
        actor_loss = None
        if update_actor:
            actor_loss = -self.critic1(s, self._convert(self.actor(s))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update all target networks
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return (critic1_loss.item() + critic2_loss.item()) / 2, None if actor_loss is None else actor_loss.item(), (
            float(torch.min(q1, q2).mean().item())
        )


def train(args: argparse.Namespace):
    env = make_env(args.env, args.gamma, args.normalize)()
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(env.action_space, gym.spaces.Box), "TD3 only supports continuous action spaces"

    obs_shape = env.observation_space.shape
    action_dim = int(env.action_space.shape[0])
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    action_center = (high + low) / 2.0
    action_scale = (high - low) / 2.0

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    buffer = ReplayBuffer(args.buffer_size)
    agent = TD3Agent(obs_shape, action_dim, action_center, action_scale, args)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    start_time = time.time()
    last_log_step = 0

    while global_step < args.total_timesteps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add((obs, action, reward, next_obs, terminated))
        obs = next_obs
        global_step += 1

        # log episode return and episode length
        if "episode" in info:
            episode_reward = float(info["episode"]["r"])
            episode_length = int(info["episode"]["l"])
            writer.add_scalar("charts/return", episode_reward, global_step)
            writer.add_scalar("charts/length", episode_length, global_step)
            print(f"global_step={global_step}, episodic_return={episode_reward:.3f}")

            obs, _ = env.reset()

        # log steps per second
        if global_step - last_log_step >= 2000:
            sps = global_step / (time.time() - start_time)
            writer.add_scalar("charts/sps", sps, global_step)
            print(f"SPS={int(sps)}")
            last_log_step = global_step

        # optimize the model
        if global_step >= args.learning_starts and global_step % args.train_freq == 0:
            critic_loss, actor_loss, mean_q = agent.update(
                buffer.sample(args.batch_size), global_step % args.policy_freq == 0
            )
            if global_step % 100 == 0:
                writer.add_scalar("loss/critic_loss", critic_loss, global_step)
                writer.add_scalar("loss/q_value", mean_q, global_step)
                if actor_loss is not None:
                    writer.add_scalar("loss/actor_loss", actor_loss, global_step)

    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)