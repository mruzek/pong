import copy
from dataclasses import dataclass
import random
from typing import Tuple, NewType, TypedDict, List, Dict
from q_learning.q_learning import QL
from q_learning.types import QLCfg, State
from collections import deque

# TODO: WHAT IS THIS AND REIMPLEMENT 

# Experience tuple type
Experience = Tuple[State, str, float, State, bool]  # (state, action, reward, next_state, done)

class DQLWithExperience(QL):
    def __init__(self, cfg: QLCfg, buffer_size: int = 10000, batch_size: int = 32):
        super().__init__(cfg)
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size) # deque O(1) vs array O(n) on pop from the front
        self.batch_size = batch_size

    def training_loop(self):
        state: State = (0, 0)  # reset position
        done = False  # finish flag

        while not done:
            # Action selection (unchanged)
            if random.random() < self.cfg.epsilon:
                action = random.choice(self.actions)
            else:
                action = max(self.q_table[state], key=self.q_table[state].get)
                max_val = max(self.q_table[state].values())
                best_actions = [a for a, v in self.q_table[state].items() if abs(v - max_val) < 1e-10]
                action = random.choice(best_actions)

            # Environment interaction
            next_state: State = self.get_next_state(state, action)
            reward = 1 if next_state == self.cfg.goal else self.cfg.step_reward
            terminal = next_state == self.cfg.goal # NOTE

            # Store experience in replay buffer
            self.replay_buffer.append((state, action, reward, next_state, terminal))

            # Learn from experience replay
            self.learn_from_experience()
            
            # Move to next state
            state = next_state
            
            # Check if done
            if state == self.cfg.goal:
                done = True
                
        # Decay epsilon (unchanged)
        self.cfg.epsilon = max(self.cfg.min_epsilon, self.cfg.epsilon * self.cfg.epsilon_decay)

    def learn_from_experience(self):
        # Skip if not enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Update Q-values for each experience in batch
        for state, action, reward, next_state, done in batch:
            # Q-learning update (unchanged formula)
            self.q_table[state][action] = self.q_table[state][action] + self.cfg.alpha * (
                reward + (0 if done else self.cfg.gamma * max(self.q_table[next_state].values())) - 
                self.q_table[state][action]
            )