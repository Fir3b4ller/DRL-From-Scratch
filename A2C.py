import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id: str):
    def thunk() -> gym.Env:
        return gym.make(env_id)
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2C")
    parser.add_argument("--exp_name", type=str, default="A2C")
    parser.add_argument("--env", type=str, default="CartPole-v1") # CartPole-v1, LunarLander-v2, Acrobot-v1
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--anneal_lr", type=bool, default=False)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    return parser.parse_args()


class ActorNetwork(nn.Module):
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


class A2CAgent:
    def __init__(self, obs_shape, action_dim: int, args: argparse.Namespace):
        self.gamma = args.gamma
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.gae_lambda = args.gae_lambda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = int(np.array(obs_shape).prod())
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr
        )

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().cpu().numpy()

    def set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def update(self, obs, actions, rewards, dones, last_obs):
        num_steps, num_envs = obs.shape[0], obs.shape[1]
        s = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        r = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device)

        logits = self.actor(s)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(a)
        entropy = dist.entropy().mean()
        values = self.critic(s)

        with torch.no_grad():
            s_last = torch.as_tensor(np.asarray(last_obs), dtype=torch.float32, device=self.device)
            next_value = self.critic(s_last)
            advantage = torch.zeros_like(r)
            gae = torch.zeros_like(next_value)
            for t in reversed(range(num_steps)):
                v_next = next_value if t == num_steps - 1 else values[t + 1]
                delta = r[t] + self.gamma * v_next * (1.0 - done[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done[t]) * gae
                advantage[t] = gae
            returns = (advantage + values).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * (returns - values).pow(2).mean()
        loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()


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

    agent = A2CAgent(obs_shape, action_dim, args)

    writer = SummaryWriter(f"runs/{run_name}")
    # log hyperparameters
    hparams_rows = ["| parameters | value |", "|---|---|"] + \
        [f"| {k} | {v} |" for k, v in vars(args).items()]
    writer.add_text("hyperparameters", "\n".join(hparams_rows), global_step=0)

    ep_rewards = [[] for _ in range(args.num_envs)]

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    last_step, last_time = 0, time.time()
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

            for i in range(args.num_envs):
                ep_rewards[i].append(rewards[i])

            obs = next_obs
            global_step += args.num_envs

            for i in np.flatnonzero(dones):
                episode_reward = sum(ep_rewards[i])
                episode_length = len(ep_rewards[i])
                writer.add_scalar("charts/return", episode_reward, global_step)
                writer.add_scalar("charts/length", episode_length, global_step)
                print(f"global_step={global_step}, episodic_return={episode_reward}")
                ep_rewards[i].clear()

            if global_step - last_log_step >= 2000:
                delta_steps = global_step - last_step
                elapsed = time.time() - last_time
                sps = delta_steps / elapsed
                writer.add_scalar("charts/sps", sps, global_step)
                print(f"SPS={int(sps)}")
                last_step, last_time = global_step, time.time()
                last_log_step = global_step

        # update
        if args.anneal_lr:
            frac = 1.0 - global_step / args.total_timesteps
            lr_now = args.lr * frac
            agent.set_lr(lr_now)
            writer.add_scalar("charts/learn_rate", lr_now, global_step)

        actor_loss, critic_loss, entropy = agent.update(
            np.stack(rollout_obs),
            np.stack(rollout_actions),
            np.stack(rollout_rewards),
            np.stack(rollout_dones),
            obs,
        )
        writer.add_scalar("loss/actor_loss", actor_loss, global_step)
        writer.add_scalar("loss/critic_loss", critic_loss, global_step)
        writer.add_scalar("loss/entropy", entropy, global_step)

    writer.close()
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)