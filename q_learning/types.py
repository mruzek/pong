from dataclasses import dataclass
from typing import Tuple, TypedDict
#import inspect

State = Tuple[int, int]

@dataclass
class QLCfg:                  # fuck your defaults
    grid_size: int            # grid_size: int = 20
    goal: State               # goal: State = (18, 18)
    step_reward: int          # step_reward: int = 0
    alpha: float              # alpha: float = 0.1
    gamma: float              # gamma: float = 0.9
    epsilon: float            # epsilon: float = 1.0
    epsilon_decay: float      # epsilon_decay: float = 0.99 # slower would be sth like 0.999 or 0.9995
    min_epsilon: float        # min_epsilon: float = 0.01
    episodes: int             # episodes: int = 100 #low to test fails
    q_delta_threshold: float  # q_delta_threshold: float = None # default by episodes
    learning_rate: float      # learning_rate: float = 0.1

class Actions(TypedDict):
    u: float
    d: float
    l: float
    r: float

    def __str__(self):
        return f"u: {self.u:.4f}, d: {self.d:.4f}, l: {self.l:.4f}, r: {self.r:.4f}"
