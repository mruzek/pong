from dataclasses import dataclass
import inspect
import random
import sys
from typing import Tuple, NewType, TypedDict
import numpy as np
from numba import njit, jit
import time

# Keep original typings for compatibility
@dataclass
class QLCfg:
    grid_size: int = 20
    goal: Tuple[int, int] = (18, 18)
    step_reward: int = 0
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    episodes: int = 100  # low to test fails
    learning_rate: float = 0.1

class Actions(TypedDict):
    u: float
    d: float
    l: float
    r: float

    def __str__(self):
        return f"u: {self.u:.4f}, d: {self.d:.4f}, l: {self.l:.4f}, r: {self.r:.4f}"

State = Tuple[int, int]

# Pre-compute action values for faster lookup
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_MAP = {
    'u': ACTION_UP,
    'd': ACTION_DOWN,
    'l': ACTION_LEFT,
    'r': ACTION_RIGHT,
}
ACTION_CHARS = ('u', 'd', 'l', 'r')

# Optimized state transition function with Numba
@njit
def get_next_state_fast(state_x, state_y, action, grid_size):
    if action == ACTION_UP and state_y > 0:
        return state_x, state_y - 1
    if action == ACTION_DOWN and state_y < grid_size - 1:
        return state_x, state_y + 1
    if action == ACTION_LEFT and state_x > 0:
        return state_x - 1, state_y
    if action == ACTION_RIGHT and state_x < grid_size - 1:
        return state_x + 1, state_y
    return state_x, state_y

# JIT-compiled training loop for maximum performance
@njit
def train_fast(q_table, grid_size, goal_x, goal_y, episodes, alpha, gamma, 
               epsilon, epsilon_decay, min_epsilon, step_reward):
    for episode in range(episodes):
        state_x, state_y = 0, 0  # reset position
        done = False  # finish flag
        
        while not done:
            # Epsilon-greedy action selection (optimized)
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)  # random choice of 4 actions
            else:
                # Get action with maximum Q-value
                state_actions = q_table[state_x, state_y]
                max_value = np.max(state_actions)
                # Find all actions with the max value (within a small numerical tolerance)
                max_actions = np.where(np.abs(state_actions - max_value) < 1e-10)[0]
                # Randomly choose from the best actions
                action = max_actions[np.random.randint(0, len(max_actions))]
                
            # Get next state
            next_state_x, next_state_y = get_next_state_fast(state_x, state_y, action, grid_size)
            
            # Reward system
            reward = 1 if (next_state_x == goal_x and next_state_y == goal_y) else step_reward
            
            # Q-value update (Bellman equation)
            next_max_q = np.max(q_table[next_state_x, next_state_y])
            current_q = q_table[state_x, state_y, action]
            
            # Update Q-value
            q_table[state_x, state_y, action] = current_q + alpha * (
                reward + gamma * next_max_q - current_q
            )
            
            # Move to next state
            state_x, state_y = next_state_x, next_state_y
            
            # Check if goal reached
            if state_x == goal_x and state_y == goal_y:
                done = True
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    return q_table, epsilon

class QLOptimized:
    def __init__(self, cfg: QLCfg):
        self.cfg = cfg
        self.actions = ('u', 'd', 'l', 'r')
        
        # Use numpy array instead of dictionary for Q-table (much faster access)
        self.q_table = np.zeros((self.cfg.grid_size, self.cfg.grid_size, 4), dtype=np.float32)
        
        # For compatibility with original interface
        self._dict_qtable = None
        
    def _update_dict_qtable(self):
        """Convert numpy Q-table to dictionary format for compatibility with original interface"""
        self._dict_qtable = {}
        for x in range(self.cfg.grid_size):
            for y in range(self.cfg.grid_size):
                self._dict_qtable[(x, y)] = {
                    'u': self.q_table[x, y, ACTION_UP],
                    'd': self.q_table[x, y, ACTION_DOWN],
                    'l': self.q_table[x, y, ACTION_LEFT],
                    'r': self.q_table[x, y, ACTION_RIGHT]
                }
    
    def train(self):
        goal_x, goal_y = self.cfg.goal
        
        # Call the JIT-compiled training function
        self.q_table, self.cfg.epsilon = train_fast(
            self.q_table, self.cfg.grid_size, goal_x, goal_y, 
            self.cfg.episodes, self.cfg.alpha, self.cfg.gamma,
            self.cfg.epsilon, self.cfg.epsilon_decay, self.cfg.min_epsilon, 
            self.cfg.step_reward
        )
        
        # Update dictionary representation after training
        self._update_dict_qtable()
    
    def run(self, start: State = None, goal: State = None):
        # Use the dictionary format for compatibility with original interface
        if self._dict_qtable is None:
            self._update_dict_qtable()
            
        if start is None:
            start = (0, 0)
        if goal is None:
            goal = self.cfg.goal
            
        state: State = start
        visited_states = set()
        
        while True:
            visited_states.add(state)
            
            # Get action using dictionary interface for compatibility
            action = max(self._dict_qtable[state], key=self._dict_qtable[state].get)
            state = self.get_next_state(state, action)
            
            if state in visited_states:
                return 0
            if state == self.cfg.goal:
                return 1
    
    def get_next_state(self, state: Tuple[int, int], action) -> State:
        # This is kept for compatibility with the original interface
        action_idx = ACTION_MAP.get(action)
        next_x, next_y = get_next_state_fast(state[0], state[1], action_idx, self.cfg.grid_size)
        return (next_x, next_y)
