import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import random
import time
from numba import jit
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

@dataclass
class QLearnConfig:
    grid_size: int = 50
    goal: Tuple[int, int] = (29, 42)
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    episodes: int = 1000
    learning_rate: float = 0.1

@jit(nopython=True)
def _get_max_q_value(q_values: np.ndarray) -> float:
    """Get maximum Q-value using Numba optimization."""
    return np.max(q_values)

class OptimizedQLearning:
    def __init__(self, config: QLearnConfig):
        self.config = config
        self.actions = ['up', 'down', 'left', 'right']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        # Initialize Q-table with small random values for better exploration
        self.Q_table = np.random.uniform(0, 0.1, 
                                       (config.grid_size, config.grid_size, len(self.actions))
                                       )
        
        self.epsilon = config.epsilon
        self.transitions = self._precompute_transitions()
        
    def _precompute_transitions(self) -> Dict[Tuple[int, int, str], Tuple[int, int]]:
        """Precompute all possible state transitions."""
        transitions = {}
        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                for action in self.actions:
                    next_x, next_y = x, y
                    if action == 'up' and y > 0:
                        next_y = y - 1
                    elif action == 'down' and y < self.config.grid_size - 1:
                        next_y = y + 1
                    elif action == 'left' and x > 0:
                        next_x = x - 1
                    elif action == 'right' and x < self.config.grid_size - 1:
                        next_x = x + 1
                    transitions[(x, y, action)] = (next_x, next_y)
        return transitions

    def get_next_state(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next state using precomputed transitions."""
        return self.transitions[(state[0], state[1], action)]

    def get_action(self, state: Tuple[int, int], explore: bool = True) -> str:
        """Get action using epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = self.Q_table[state[0], state[1]]
        return self.actions[np.argmax(q_values)]

    def update_q_value(self, state: Tuple[int, int], action: str, 
                      next_state: Tuple[int, int], reward: float) -> None:
        """Update Q-value using the Bellman equation."""
        action_idx = self.action_to_idx[action]
        current_q = self.Q_table[state[0], state[1], action_idx]
        next_max_q = _get_max_q_value(self.Q_table[next_state[0], next_state[1]])
        
        # Q-learning update
        self.Q_table[state[0], state[1], action_idx] = current_q + \
            self.config.alpha * (reward + self.config.gamma * next_max_q - current_q)

    def train(self) -> List[int]:
        """Train the agent."""
        episode_lengths = []
        start_time = time.time()
        
        for episode in range(self.config.episodes):
            state = (0, 0)
            steps = 0
            visited_states = set()
            
            while state != self.config.goal and steps < self.config.grid_size * 4:
                visited_states.add(state)
                action = self.get_action(state)
                next_state = self.get_next_state(state, action)
                
                # Reward shaping: encourage exploration and shortest path
                reward = 0
                if next_state == self.config.goal:
                    reward = 1.0
                elif next_state in visited_states:  # Penalty for revisiting states
                    reward = -0.1
                else:
                    # Small reward for moving closer to goal
                    current_dist = abs(state[0] - self.config.goal[0]) + abs(state[1] - self.config.goal[1])
                    next_dist = abs(next_state[0] - self.config.goal[0]) + abs(next_state[1] - self.config.goal[1])
                    if next_dist < current_dist:
                        reward = 0.01
                
                self.update_q_value(state, action, next_state, reward)
                state = next_state
                steps += 1
            
            episode_lengths.append(steps)
            self.epsilon = max(self.config.min_epsilon, 
                             self.epsilon * self.config.epsilon_decay)
            
            if episode % 100 == 0:
                avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                print(f"Episode {episode}, Avg Length: {avg_length:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}")

        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        return episode_lengths

    def test(self) -> List[Tuple[int, int]]:
        """Test the trained agent."""
        path = []
        state = (0, 0)
        visited = set()
        max_steps = self.config.grid_size * 2
        steps = 0
        
        while state != self.config.goal and state not in visited and steps < max_steps:
            path.append(state)
            visited.add(state)
            action = self.get_action(state, explore=False)
            state = self.get_next_state(state, action)
            steps += 1
        
        if state == self.config.goal:
            path.append(state)
        
        return path

#visualizer
def visualize_path(grid_size, path):
    plt.figure(figsize=(6,6))
    plt.grid(True)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    path = np.array(path)
    plt.plot(path[:,1], path[:,0], 'b-o', label='Path')
    plt.plot(path[0,1], path[0,0], 'go', label='Start', markersize=12)
    plt.plot(path[-1,1], path[-1,0], 'ro', label='End', markersize=12)    
    plt.legend()
    plt.show()

if __name__ == "__main__":
  
    _goal = (899, 899)
    _grid = 1000
    config = QLearnConfig(
        episodes=100000, # 1949 on 100k with e 1
        grid_size=_grid,
        goal=_goal,
        epsilon=10000,
        epsilon_decay=0.999,  # Slower decay for better exploration
        min_epsilon=0.05, #0.01,
        alpha=0.1, #0.1,
        gamma=0.99  # Higher discount factor for longer-term planning
    )
    # NOTE: large epsilon VS slow decay ?
     
    agent = OptimizedQLearning(config)
    print("Training Q-learning agent...")
    episode_lengths = agent.train()
    path = agent.test()

    if path[-1] == _goal:
      print(f"\nOptimal path length: {len(path)}")
      print("Path:", path)
      visualize_path(_grid, path)
    else:
      print(f"Goal not reached {path[-1]}")

# TODO: 1
# If it quickly finds a decent path (~170 steps) but never improves,
# it might be because the Q-values aren't differentiating enough between slightly better and slightly worse paths.
# Possible Fix: Try lowering the learning rate (α) over time so it fine-tunes instead of making big updates.
# Another Fix: Use soft updates instead of fully overriding Q-values:
#   Setting α closer to 0 over time (e.g., start at 0.1, decay to 0.01) can help refine the best paths.

