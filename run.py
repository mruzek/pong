import time
from q_learning.deep_q_learning import DQL
from q_learning.q_learning import QL
from q_learning.q_learning_optimized import QLOptimized
from q_learning.types import QLCfg

config_default = QLCfg(
    grid_size = 20,
    goal = (18, 18),
    step_reward = 0,
    alpha = 0.1,
    gamma = 0.9,
    epsilon = 1.0,
    epsilon_decay = 0.99,
    min_epsilon = 0.01,
    episodes = 100,
    q_delta_threshold = None,
    learning_rate = 0.1
)

#config_qdt = QLCfg(q_delta_threshold = 0.03)

agent = QL(config_default)
agent_o = QLOptimized(config_default)
agent_d = DQL(config_default)

# THE VERY BASIC RUN

# QL
start_time = time.time()
agent.train()
elapsed_time = time.time() - start_time
print(f"QL Training time: {elapsed_time} s")

# Optimized
start_time = time.time()
agent_o.train()
elapsed_time = time.time() - start_time
print(f"Optimized QL Training time: {elapsed_time} s")

#DQL
start_time = time.time()
agent_d.train()
elapsed_time = time.time() - start_time
print(f"DQL Training time: {elapsed_time} s")

print(f"QL: {agent.run()}")
print(f"Optimized: {agent_o.run()}")
print(f"DQL: {agent_d.run()}")

#print(f"(5, 5): {agent.run(start=(5, 5))}")
# for x in range(20):
#    for y in range(20):
#            print(f"({x}, {y}): {agent.run(start=(x, y))}")