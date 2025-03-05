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
