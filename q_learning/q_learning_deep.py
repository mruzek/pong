import copy
from dataclasses import dataclass
import random
from typing import Tuple, NewType, TypedDict
from q_learning.types import QLCfg, State
     
# TODO: THIS IS COPYPASTE OF QLEARNING
#       WILL BE UPDATED SOON

class DQL:
    def __init__(self, cfg: QLCfg):
        self.cfg = cfg
        self.actions = ('u', 'd', 'l', 'r')
        self.q_table = {(x, y): {'u': 0, 'd': 0, 'l': 0, 'r': 0} for x in range(self.cfg.grid_size) for y in range(self.cfg.grid_size)}
        self.q_delta = 999

    def train(self):
        if self.cfg.q_delta_threshold:
            episode = 0
            prev_q_table = copy.deepcopy(self.q_table)
            while self.q_delta > self.cfg.q_delta_threshold:
                self.training_loop()
                # Calculate maximum change in Q-values - more elegantly
                self.q_delta = max(
                    abs(self.q_table[state][action] - prev_q_table[state][action])
                    for state in self.q_table
                    for action in self.actions
                )
                prev_q_table = copy.deepcopy(self.q_table)
                print(f"Ep.{episode}: ε:{self.cfg.epsilon} | qδ: {self.q_delta}")
                episode += 1
        else:
            for episode in range(self.cfg.episodes):
                print(f"Ep.{episode}: ε:{self.cfg.epsilon}")
                self.training_loop()

    def training_loop(self):
        state: State = (0, 0) # reset position
        done = False # finish flag

        while not done:
            if random.random() < self.cfg.epsilon:
                action = random.choice(self.actions)
            else:
                action = max(self.q_table[state], key=self.q_table[state].get)
                max_val = max(self.q_table[state].values())
                best_actions = [a for a, v in self.q_table[state].items() if abs(v - max_val) < 1e-10]
                action = random.choice(best_actions)

            next_state: State = self.get_next_state(state, action)
            reward = 1 if next_state == self.cfg.goal else self.cfg.step_reward

            self.q_table[state][action] = self.q_table[state][action] + self.cfg.alpha * (
                reward + self.cfg.gamma * max(self.q_table[next_state].values()) - self.q_table[state][action]
            )
            # move to next state
            state = next_state
            # end
            if state == self.cfg.goal:
                done = True
        # decay epsilon
        self.cfg.epsilon = max(self.cfg.min_epsilon, self.cfg.epsilon * self.cfg.epsilon_decay)

    def run(self, start: State = None, goal: State = None):
        if start is None:
            start = (0, 0)
        if goal is None:
            goal = self.cfg.goal
        state: State = start
        visited_states = set()
        while True:
            print(f"{state}")
            visited_states.add(state)
            action = max(self.q_table[state], key=self.q_table[state].get)
            state = self.get_next_state(state, action)
            if state in visited_states:
                return 0
            if state == self.cfg.goal:
                print(f"length: {len(visited_states)}")
                return 1 

    def get_next_state(self, state: Tuple[int, int], action) -> State:
        # TODO: optimize
        if action == 'u' and state[1] > 0:
            return (state[0],state[1]-1)
        if action == 'd' and state[1] < self.cfg.grid_size - 1:
            return (state[0],state[1]+1)
        if action == 'l' and state[0] > 0:
            return (state[0]-1,state[1])
        if action == 'r' and state[0] < self.cfg.grid_size - 1:
            return (state[0]+1,state[1])
        return (state[0], state[1])
