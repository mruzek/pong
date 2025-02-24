from dataclasses import dataclass
import random
from typing import Tuple, NewType, TypedDict

# Q-Learning Algorithm
#
# Problem Setup:
# Imagine a 4x4 grid world where:
#   The agent starts at (0,0).
#   The goal is at (3,3).
#   The agent can move up, down, left, or right.
#   Moving into walls keeps the agent in the same position.
#   The agent gets a reward of +1 for reaching the goal and 0 otherwise.
#
# Explanation:
# 1. Q-table Initialization: Each state (x, y) has Q-values for all four possible actions.
#    Exploration vs. Exploitation:
#     Early training: Random exploration (epsilon is high).
#     Later: More exploitation (choosing the best action).
# 2. Q-value Updates:
#    Uses the Bellman equation.
#    Updates based on the best future Q-value.
# 3. Training:
#    The agent learns over 1000 episodes.
#    Epsilon decays so the agent shifts from exploring to exploiting.
# 4. Testing:
#    The agent follows the optimal path from (0,0) to (3,3).
# Results:
#    The agent learns the shortest path in the grid.
#    The Q-table stores the best actions for each state.

# typing

@dataclass
class QLCfg:
    grid_size: int = 5
    goal: Tuple[int, int] = (3, 3)
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    episodes: int = 1000
    learning_rate: float = 0.1

class Actions(TypedDict):
    u: float
    d: float
    l: float
    r: float

    def __str__(self):
        return f"u: {self.u:.4f}, d: {self.d:.4f}, l: {self.l:.4f}, r: {self.r:.4f}"

State = Tuple[int, int]
     
class QL:
    def __init__(self, cfg: QLCfg):
        self.cfg = cfg
        self.actions = ('u', 'd', 'l', 'r')
        self.q_table = {(x, y): {'u': 0.1, 'd': 0.1, 'l': 0.1, 'r': 0.1} for x in range(self.cfg.grid_size) for y in range(self.cfg.grid_size)}

    def train(self):
        # training loop
        for episode in range(self.cfg.episodes):
            state: State = (0, 0) # reset position
            done = False # finish flag

            while not done:
            # choose action    
                # epsilon greedy strategy
                # TODO: try Softmax (Boltzmann) Exploration
                # TODO: try Upper Confidence Bound
                # TODO: try Thompson Sampling
                if random.random() < self.cfg.epsilon:
                    action = random.choice(self.actions)
                else:
                    action = max(self.q_table[state], key=self.q_table[state].get)

            # do action
                next_state: State = self.get_next_state(state, action)

            # reward system
                # NOTE: this is correct because next state = goal means, correct action was chosen in current state 
                #       reward is for making action from the state, not for being in a state 
                # TODO: small penalty for each step to encourage the agent to reach the goal faster: else -0.01
                reward = 1 if next_state == self.cfg.goal else 0

            # populate Qtable = Bellman formula
                # TODO: try n step
                # TODO: try monte carlo
                # TODO: try TD(λ)
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
            print(self.cfg.epsilon)

    def run(self):
        print(f"run")
        state: State = (0, 0)
        done = False
        
        while not done:
            print(state)
            action = max(self.q_table[state], key=self.q_table[state].get)
            state = self.get_next_state(state, action)
            if state == self.cfg.goal:
                done = True    

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

if __name__ == "__main__":
    agent = QL(QLCfg())
    agent.train()
    print(agent.q_table)
    agent.run()