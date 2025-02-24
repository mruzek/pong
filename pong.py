import gymnasium

# https://github.com/openai/gym/issues/3131
# 3 days ago fuckers
# >>> pip install "gymnasium[atari,accept-rom-license]==0.29.1"

env = gymnasium.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()

total_reward = 0
step_count = 0

for _ in range(1000):  # Shorter range to make output readable
    step_count += 1
    # action = env.action_space.sample()  # 0, 2, 3

    action = 3

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    print(f"\nStep {step_count}:")
    print(f"Action taken: {action}")
    
    # Observation in Pong is a 210x160x3 RGB image array
    print(f"Observation shape: {obs.shape}")  
    print(f"Reward: {reward}")  # +1 when you score, -1 when opponent scores
    print(f"Total Reward: {total_reward}")
    print(f"Terminated: {terminated}")      
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    
    if terminated or truncated:
        print("\nEpisode ended!")
        obs, info = env.reset()
        total_reward = 0
        step_count = 0

env.close()