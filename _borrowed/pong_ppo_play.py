import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

# Import PPOAgent and PPONetwork from your existing script
from pong__ppo import PPOAgent  # Change 'your_script_name' to your actual script filename

def playtest(model_path, render_mode="human"):
    # Initialize the environment with the same preprocessing as training
    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

    state_shape = (4, 84, 84)
    n_actions = env.action_space.n

    # Create agent and load model weights
    agent = PPOAgent(state_shape, n_actions)
    agent.network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    agent.network.eval()

    state, _ = env.reset()
    state = np.array(state)

    done = False
    total_reward = 0

    while not done:
        action, _ = agent.network.get_action(state)  # Get action from trained model
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()  # Render the game
        state = np.array(next_state)
        total_reward += reward

    print(f"Test run completed. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    model_path = "pong_ppo_episode_600.pth"
    playtest(model_path)
