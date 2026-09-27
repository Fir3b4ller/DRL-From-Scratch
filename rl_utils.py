import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
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
            torch.as_tensor(a),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(s_, dtype=torch.float32),
            torch.as_tensor(done, dtype=torch.float32),
        )


def linear_schedule(start: float, end: float, total_steps: int):
    slope = (end - start) / total_steps

    def schedule(step: int) -> float:
        return max(end, start + slope * step)

    return schedule