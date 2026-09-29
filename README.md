# DRL-algorithms

A collection of PyTorch implementations for classic deep reinforcement learning (DRL) algorithms, tailored for self-study and academic research.

The project reimplements the core training loops by hand in PyTorch, without relying on RL library wrappers, drawing on the ideas from [动手学强化学习 (Easy Reinforcement Learning)](https://hrl.boyuai.com/) and [cleanrl](https://github.com/vwxyzjn/cleanrl). This makes it easy to understand and compare each algorithm.

## Implemented Algorithms

| Category | Algorithm | Typical Environments |
| --- | --- | --- |
| Value-based | DQN | CartPole-v1 / Acrobot-v1 / LunarLander-v2 |
| Value-based | Double DQN | CartPole-v1 / Acrobot-v1 / LunarLander-v2 / MountainCar-v0 |
| Value-based | Dueling DQN | CartPole-v1 / Acrobot-v1 / LunarLander-v2 |
| Policy-based | REINFORCE | CartPole-v1 / Acrobot-v1 / LunarLander-v2 |
| Policy-based | Actor-Critic | CartPole-v1 / Acrobot-v1 / LunarLander-v2 |
| Policy-based | A2C | CartPole-v1 / Acrobot-v1 / LunarLander-v2 |
| Policy-based | TRPO | CartPole-v1 / LunarLander-v2 |
| Policy-based | PPO | CartPole-v1 / LunarLander-v2 / Acrobot-v1 |
| Policy-based | PPO (continuous) | MuJoCo (Hopper-v4 / HalfCheetah-v4 / ...) / BipedalWalker-v3 |
| Policy-based | DDPG | Pendulum-v1 / LunarLanderContinuous-v2 |
| Policy-based | TD3 | Pendulum-v1 / LunarLanderContinuous-v2 / Ant-v4 |
| Policy-based | SAC | Pendulum-v1 / LunarLanderContinuous-v2 / Hopper-v4 |

## Project Structure

```
DRL-algorithms/
├── DRL_algorithms/        # Algorithm implementations (one file per algorithm)
│   ├── rl_utils.py        # Shared utilities: ReplayBuffer, linear_schedule
│   ├── DQN.py
│   ├── DDQN.py
│   ├── Dueling DQN.py
│   ├── REINFORCE.py
│   ├── Actor-Critic.py
│   ├── A2C.py
│   ├── TRPO.py
│   ├── PPO.py
│   ├── PPO_continuous.py
│   ├── DDPG.py
│   ├── TD3.py
│   └── SAC.py
├── exp_result/            # Experiment results (learning curves, PNG / SVG)
├── requirements.txt       # Dependencies
└── README.md
```


## Installation

The project is developed and tested with Python 3.9.25.

```bash
pip install -r requirements.txt
```

> Note: If you use a GPU, install a CUDA-compatible PyTorch build (the `torch==2.4.1` in `requirements.txt` is the generic/CPU build). MuJoCo environments (`Ant-v4`, `Hopper-v4`, etc.) additionally require a matching `mujoco` installation.

## Quick Start

Each algorithm script is configured through command-line arguments. For example:

```bash
# Train DQN on CartPole-v1
python DRL_algorithms/DQN.py --env CartPole-v1

# Train PPO on CartPole-v1
python DRL_algorithms/PPO.py --env CartPole-v1

# Train SAC on Pendulum-v1
python DRL_algorithms/SAC.py --env Pendulum-v1
```

Common arguments: `--env`, `--seed`, `--total_timesteps`, `--lr`, `--gamma`, etc.

Launch TensorBoard to view training curves:

```bash
tensorboard --logdir runs
```

## Experiment Results

All experiment learning curves and results are stored under `exp_result/`, organized by algorithm (e.g. `exp_result/DQN/`, `exp_result/PPO/`). Each experiment provides both PNG and SVG output.

## References

- [动手学强化学习 (Easy Reinforcement Learning)](https://hrl.boyuai.com/)
- [cleanrl](https://github.com/vwxyzjn/cleanrl)