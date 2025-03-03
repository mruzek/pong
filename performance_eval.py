import csv
from datetime import datetime
import sys
from typing import Dict, Tuple
from q_learning_optimized import QLCfg, QLOptimized
from visualize import make_chart

# TODO: BATCH TEST
# alpha
# gamma
# epsilon, decay
#   succ rate, 
#   convergence speed - once q_value_change 
#       MAX np.max(np.abs(Q_table - previous_Q_table)) < 0.001 it converged
#       AVG q_value_change = np.mean(np.abs(Q_table - previous_Q_table))
#       SUM q_value_change = np.sum(np.abs(Q_table - previous_Q_table))
#   reward stability

# TODO? Bayesian Optimization: Use libraries like Optuna or Hyperopt to find optimal parameters with fewer trials. 
# Bayesian Optimization: Use libraries like scikit-optimize to efficiently find optimal parameters.
# ------
# Alpha (Learning Rate):
#   Controls how quickly the algorithm updates Q-values based on new experiences.
#   Too high: The algorithm may overshoot optimal values.
#   Too low: The algorithm learns very slowly.
# Gamma (Discount Factor):
#   Determines the importance of future rewards.
#   Too high: The algorithm may prioritize long-term rewards too much, leading to slower convergence.
#   Too low: The algorithm becomes short-sighted and may not find the optimal policy.
# Epsilon (Exploration Rate):
#   Controls the trade-off between exploration (trying new actions) and exploitation (using known actions).
#   Too high: The algorithm explores too much and may not converge.
#   Too low: The algorithm may get stuck in suboptimal policies


def eval_alpha_gamma(config: QLCfg, alpha_range: Tuple[float, ...], gamma_range: Tuple[float, ...], name: None):
    print('\nCreate alpha gamma chart')
    print(f"α range: {alpha_range}")
    print(f"γ range: {gamma_range}")
    print("------------------------")

    res: Dict[Tuple[float, float], float] = {}
    if not name:
        name = f"./results/log_{datetime.now().timestamp()}"
    else:
        name = f"./results/log_{name}_{datetime.now().timestamp()}"
    with open(f"{name}.csv", "w") as f:
        f.write("alpha, gamma, perf\n")
    for a in alpha_range: 
        for g in gamma_range:
            config.alpha = a
            config.gamma = g
            res[(a, g)] = eval_cfg(config, 200)
            with open(f"{name}.csv", "a") as f:
                f.write(f"{a}, {g}, {res[(a, g)]}\n")
            print(f"Evaluated (α:{a}, γ:{g}) [{len(res) / (len(alpha_range) * len(gamma_range)) * 100:.2F}%]")
    make_chart(name)
    return res
    
def eval_cfg(conf: QLCfg, run_count = 100):
    succ_run_count = 0
    for run in range(run_count):
        run += 1
        #agent = QL(conf)
        agent = QLOptimized(conf)
        agent.train()
        succ_run_count += agent.run()
    return succ_run_count / run_count

if __name__ == "__main__":
    # config_default = QLCfg(
    #     grid_size = 20,
    #     goal = (18, 18),
    #     step_reward = 0,
    #     alpha = 0.1,
    #     gamma = 0.9,
    #     epsilon = 1.0,
    #     epsilon_decay = 0.99,
    #     min_epsilon = 0.01,
    #     episodes = 1000,
    #     learning_rate = 0.1
    # )

    # alpha, gamma
    alpha_range = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    gamma_range = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    static_epsilon_range = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    epsilon_decay_range = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999)
    learning_rate_range = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)

    # static epsilon 100 episodes
    # for e in static_epsilon_range:
    #     config = QLCfg(
    #         epsilon = e,
    #         epsilon_decay = 1,
    #         episodes = 100
    #     )
    #     eval_alpha_gamma(config, alpha_range, gamma_range, "def_static_epsilon_100_episodes")

    # # epsilon decay 100 episodes
    # for ed in epsilon_decay_range:
    #     config = QLCfg(
    #         epsilon_decay = ed,
    #         episodes = 100
    #     )
    #     eval_alpha_gamma(config, alpha_range, gamma_range, "def_epsilon_decay_100_episodes")

    # # learning rate 100 episodes
    # for e in learning_rate_range:
    #     config = QLCfg(
    #         epsilon = e,
    #         epsilon_decay = 1,
    #         episodes = 100
    #     )
    #     eval_alpha_gamma(config, alpha_range, gamma_range, "def_learning_rate_100_episodes")

# TODO: MAKE TRAINING SO IT SAVES LOWER EPISODES VERSION ALONG THE WAY
