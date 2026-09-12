import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """固定容量经验回放缓冲区。

    每条 transition 形如 (s, a, r, s', done)。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, done = map(np.stack, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.long),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(s_, dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32),
        )


def linear_schedule(start: float, end: float, total_steps: int):
    """返回因变量随 step 从 start 线性降至 end 的调度函数。

    参数 step 需满足 0 <= step <= total_steps，结果截断到不小于 end。
    """
    slope = (end - start) / total_steps

    def schedule(step: int) -> float:
        return max(end, start + slope * step)

    return schedule