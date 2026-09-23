import argparse
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter


def make_env(env_id: str):
    def thunk() -> gym.Env:
        return gym.make(env_id)
    return thunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRPO")
    parser.add_argument("--exp_name", type=str, default="TRPO")
    parser.add_argument("--env", type=str, default="LunarLander-v2") # CartPole-v1, LunarLander-v2, Acrobot-v1
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    # TRPO specific
    parser.add_argument("--kl_limit", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--cg_iters", type=int, default=10)
    parser.add_argument("--backtrack_iters", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.8)
    # value function
    parser.add_argument("--vf_lr", type=float, default=5e-4)
    parser.add_argument("--vf_epochs", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
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


class TRPOAgent:
    def __init__(self, obs_shape, action_dim: int, args: argparse.Namespace):
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.kl_limit = args.kl_limit
        self.damping = args.damping
        self.cg_iters = args.cg_iters
        self.backtrack_iters = args.backtrack_iters
        self.alpha = args.alpha
        self.vf_epochs = args.vf_epochs
        self.max_grad_norm = args.max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = int(np.array(obs_shape).prod())
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.vf_lr)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().cpu().numpy()

    def flat_grad(self, y: torch.Tensor, params, retain_graph=False, create_graph=False):
        grads = torch.autograd.grad(y, params, retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([g.reshape(-1) for g in grads])

    # fisher vector product
    def fvp(self, vector, obs, dist_old, damping=True):
        logits = self.actor(obs)
        dist_new = torch.distributions.Categorical(logits=logits)
        kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()
        grad_kl = self.flat_grad(kl, self.actor.parameters(), retain_graph=True, create_graph=True)
        grad_kl_v = (grad_kl * vector).sum()
        fvp = self.flat_grad(grad_kl_v, self.actor.parameters())
        return fvp + (self.damping * vector if damping else 0)

    def conjugate_gradient(self, fvp_fn, grad):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(self.cg_iters):
            z = fvp_fn(p)
            alpha = r_dot_old / (torch.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r, r)
            if r_dot_new < 1e-10:
                break
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def line_search(self, obs, action, advantage, old_log_prob, dist_old, old_surrogate, max_vec, old_params):
        for i in range(self.backtrack_iters):
            step = max_vec * (self.alpha ** i)
            vector_to_parameters(old_params + step, self.actor.parameters())
            with torch.no_grad():
                logits = self.actor(obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)
                new_surrogate = (ratio * advantage).mean().item()
                kl = torch.distributions.kl.kl_divergence(dist_old, dist).mean().item()
            if new_surrogate > old_surrogate and kl < self.kl_limit:
                return
        vector_to_parameters(old_params, self.actor.parameters())

    def update(self, obs, actions, rewards, dones, last_obs):
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        obs_last = torch.as_tensor(np.asarray(last_obs), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        reward = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device)
        num_steps = obs.shape[0]

        # GAE advantage
        with torch.no_grad():
            values = self.critic(obs)
            next_value = self.critic(obs_last)
            advantage = torch.zeros_like(reward)
            gae = torch.zeros_like(next_value)
            for t in reversed(range(num_steps)):
                v_next = next_value if t == num_steps - 1 else values[t + 1]
                delta = reward[t] + self.gamma * v_next * (1.0 - done[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done[t]) * gae
                advantage[t] = gae
            returns = (advantage + values).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        logits_old = self.actor(obs).detach()
        dist_old = torch.distributions.Categorical(logits=logits_old)
        old_log_prob = dist_old.log_prob(action).detach()
        entropy = dist_old.entropy().mean()

        # compute natural gradient direction
        logits_now = self.actor(obs)
        dist_now = torch.distributions.Categorical(logits=logits_now)
        now_log_prob = dist_now.log_prob(action)
        actor_params = list(self.actor.parameters())
        old_params = parameters_to_vector(actor_params).detach()
        surrogate = (torch.exp(now_log_prob - old_log_prob) * advantage).mean()
        grad = self.flat_grad(surrogate, actor_params, retain_graph=True)
        fvp = lambda v: self.fvp(v, obs, dist_old)
        step_dir = self.conjugate_gradient(fvp, grad)

        xTx = torch.dot(step_dir, self.fvp(step_dir, obs, dist_old, damping=False))
        max_step_len = torch.sqrt(2.0 * self.kl_limit / (xTx + 1e-8))
        old_surrogate = surrogate.item()

        # line search
        self.line_search(obs, action, advantage, old_log_prob, dist_old, old_surrogate, max_step_len * step_dir, old_params)

        with torch.no_grad():
            logits_new = self.actor(obs)
            dist_new = torch.distributions.Categorical(logits=logits_new)
            kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean().item()

        # value function update
        for _ in range(self.vf_epochs):
            value_pred = self.critic(obs)
            value_loss = 0.5 * (value_pred - returns).pow(2).mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        return old_surrogate, kl, entropy.item(), value_loss.item()


def train(args: argparse.Namespace):
    envs = gym.vector.AsyncVectorEnv([make_env(args.env) for _ in range(args.num_envs)])
    run_name = f"{args.env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only supports discrete action spaces"

    obs_shape = envs.single_observation_space.shape
    action_dim = int(envs.single_action_space.n)

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    agent = TRPOAgent(obs_shape, action_dim, args)

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
        surrogate, kl, entropy, value_loss = agent.update(
            np.stack(rollout_obs),
            np.stack(rollout_actions),
            np.stack(rollout_rewards),
            np.stack(rollout_dones),
            obs,
        )
        writer.add_scalar("loss/surrogate", surrogate, global_step)
        writer.add_scalar("loss/kl", kl, global_step)
        writer.add_scalar("loss/entropy", entropy, global_step)
        writer.add_scalar("loss/value_loss", value_loss, global_step)

    writer.close()
    envs.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)