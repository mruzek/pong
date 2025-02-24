import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

class PPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPONetwork, self).__init__()
        
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.actor(conv_out), self.critic(conv_out)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

class PPOAgent:
    def __init__(self, state_shape, n_actions):
        self.network = PPONetwork(state_shape, n_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.00025)
        
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # PPO specific parameters
        self.ppo_epochs = 4
        self.batch_size = 32
        self.gamma = 0.99
        self.gae_lambda = 0.95
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # Compute values and advantages
        with torch.no_grad():
            _, values = self.network(states)
            values = values.squeeze()
            next_values = torch.cat([values[1:], values[-1:]])
        
        advantages = self.compute_gae(rewards, values.numpy(), 
                                    next_values.numpy(), dones)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Generate random mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current action probabilities and values
                action_logits, value_preds = self.network(batch_states)
                dist = Categorical(logits=action_logits)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and surrogate loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Compute actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value_preds.squeeze(), batch_returns)
                
                # Combined loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def train():
    # Create environment with wrappers
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    
    agent = PPOAgent(state_shape, n_actions)
    episodes = 10000
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state)
        
        # Storage for episode data
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Get action from policy
            action, log_prob = agent.network.get_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = np.array(next_state)
            total_reward += reward
            step_count += 1
            
            # Update every 2048 steps or at end of episode
            if step_count % 2048 == 0 or done:
                agent.update(states, actions, log_probs, rewards, dones)
                states, actions, rewards, log_probs, dones = [], [], [], [], []
            
            if step_count % 100 == 0:
                print(f"Episode: {episode}, Step: {step_count}, Reward: {total_reward}")
        
        # Save model periodically
        if episode % 100 == 0:
            torch.save(agent.network.state_dict(), f"pong_ppo_episode_{episode}.pth")
        
        print(f"Episode {episode} finished. Total Reward: {total_reward}, Steps: {step_count}")
    
    env.close()

if __name__ == "__main__":
    train()