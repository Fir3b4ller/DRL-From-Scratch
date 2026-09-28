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
        if normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def parse_args():
    parser = argparse.ArgumentParser(description="SAC")
    parser.add_argument("--exp_name", type=str, default="SAC")
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2")
    # Pendulum-v1, LunarLanderContinuous-v2, BipedalWalker-v3, Walker2d-v4, HalfCheetah-v4, Ant-v4, Swimmer-v4, Hopper-v4
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto_tune_alpha", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_starts", type=int, default=25000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--target_network_frequency", type=int, default=1)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return self.mean(h), log_std

    def get_action(self, x: torch.Tensor):
        """reparameterized sample"""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        action = torch.tanh(u)
        # Jacobian correction
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-7).sum(-1)
        return action, log_prob


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


class SACAgent:
    def __init__(self, obs_shape, action_dim: int, action_low, action_high, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_center = (self.action_low + self.action_high) / 2.0
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.gamma = args.gamma
        self.tau = args.tau
        self.auto_tune_alpha = args.auto_tune_alpha
        self.target_entropy = -action_dim

        obs_dim = int(np.array(obs_shape).prod())
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=args.q_lr)
        self.policy_frequency = args.policy_frequency
        self.target_network_frequency = args.target_network_frequency
        # entropy coefficient
        if self.auto_tune_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(args.alpha), dtype=torch.float32, device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        u = self.actor.get_action(obs_t)[0]
        action = self.action_center + u * self.action_scale
        return action.cpu().numpy().squeeze(0)

    def update(self, batch, update_actor: bool, update_target: bool):
        s, a, r, s_, done = [t.to(self.device) for t in batch]
        alpha = self.log_alpha.exp()

        # critic
        with torch.no_grad():
            next_a, next_log_prob = self.actor.get_action(s_)
            next_a_scaled = self.action_center + next_a * self.action_scale
            next_q = torch.min(self.target_critic1(s_, next_a_scaled), self.target_critic2(s_, next_a_scaled))
            target_q = r + self.gamma * (1.0 - done) * (next_q - alpha * next_log_prob)

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor + alpha
        actor_loss = alpha_loss = None
        if update_actor:
            pi_a, pi_log_prob = self.actor.get_action(s)
            pi_a_scaled = self.action_center + pi_a * self.action_scale
            actor_loss = (alpha.detach() * pi_log_prob - torch.min(self.critic1(s, pi_a_scaled), self.critic2(s, pi_a_scaled))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.auto_tune_alpha:
                alpha_loss = (-alpha * (pi_log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # soft update
        if update_target:
            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return critic_loss.item(), actor_loss, float(torch.min(q1, q2).mean().item()), alpha_loss


def train(args: argparse.Namespace):
    env = make_env(args.env, args.gamma, args.normalize)()
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(env.action_space, gym.spaces.Box), "SAC only supports continuous action spaces"

    obs_shape = env.observation_space.shape
    action_space = env.action_space
    action_dim = int(action_space.shape[0])
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    buffer = ReplayBuffer(args.buffer_size)
    agent = SACAgent(obs_shape, action_dim, low, high, args)

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
        if global_step < args.learning_starts:
            action = action_space.sample()  # random policy for initial exploration
        else:
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
            update_actor = global_step % agent.policy_frequency == 0
            update_target = global_step % agent.target_network_frequency == 0
            critic_loss, actor_loss, mean_q, alpha_loss = agent.update(buffer.sample(args.batch_size), update_actor, update_target)
            if global_step % 100 == 0:
                writer.add_scalar("loss/critic_loss", critic_loss, global_step)
                writer.add_scalar("loss/q_value", mean_q, global_step)
                if actor_loss is not None:
                    writer.add_scalar("loss/actor_loss", actor_loss.item(), global_step)
                if alpha_loss is not None:
                    writer.add_scalar("loss/alpha_loss", alpha_loss.item(), global_step)
                    writer.add_scalar("charts/alpha", agent.log_alpha.exp().item(), global_step)

    writer.close()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)