import csv
import sys
from typing import Dict, Tuple
from q_learning import QL, QLCfg

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

def eval_alpha_gamma(alpha_range: Tuple[float, ...], gamma_range: Tuple[float, ...]):
    res: Dict[Tuple[float, float], float] = {}
    for a in alpha_range: 
        for g in gamma_range:
            res[(a, g)] = eval_cfg(QLCfg(alpha = a, gamma = g), run_count = 500)
            print(res[(a, g)])
            with open("log.txt", "a") as f:
                f.write(f"({a}, {g}): {res[(a, g)]}\n")
    return res
    
def eval_cfg(conf: QLCfg, run_count = 100):
    succ_run_count = 0
    for run in range(run_count):
        run += 1
        sys.stdout.write(f"\rEvaluating {run} of {run_count} [{(run) / run_count * 100:.1f}%]")
        sys.stdout.flush()
        agent = QL(conf)
        agent.train()
        succ_run_count += agent.run()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return succ_run_count / run_count

if __name__ == "__main__":
    # config_default = QLCfg(
    #     grid_size = 20,
    #     goal = (18, 18),
    #     step_penalty = 0,
    #     alpha = 0.1,
    #     gamma = 0.9,
    #     epsilon = 1.0,
    #     epsilon_decay = 0.99,
    #     min_epsilon = 0.01,
    #     episodes = 1000,
    #     learning_rate = 0.1
    # )

    # alpha, gamma
    alpha_range = (0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.99, 0.999)
    gamma_range = (0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.99, 0.999)
    range_mini = (0.1, 0.5, 0.9)
 
    print(eval_alpha_gamma(alpha_range, gamma_range))
    print("this was run with -0.01 step reward")
    # TODO: 21st run ends on some function because 'd': Inf, also all keys are HUGE line +300

    # caffeinate -i python your_script.py

