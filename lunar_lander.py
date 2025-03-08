from typing import Set
import gymnasium as gym

# actions:                            # observation space:
    # 0: Do nothing                       # x position
    # 1: Fire left engine                 # y position
    # 2: Fire main engine (bottom)        # x velocity
    # 3: Fire right engine                # y velocity
                                          # angle
                                          # angular velocity
                                          # Left leg contact (boolean)
                                          # Right leg contact (boolean)

# Create and reset
env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

try_count = 20
for try_number in range(try_count):
    terminated = False
    step_count = 0

    while not terminated:
        step_count += 1
        # action = env.action_space.sample()
        action = 3


        observation, reward, terminated, truncated, info = env.step(action)

    print(F"Try {try_number + 1} steps: {step_count}")
    step_count = 0
    observation, info = env.reset()

env.close()