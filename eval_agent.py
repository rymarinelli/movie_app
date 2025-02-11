import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from deployment_env import DeploymentEnv 

# Load the trained PPO agent 
model = PPO.load("deployment_scaling_agent.zip")


env = DeploymentEnv(base_url="http://127.0.0.1:5000", realistic_usage=True, wait_time=1)
vec_env = DummyVecEnv([lambda: env])

# Set the environment for the loaded model.
model.set_env(vec_env)

n_episodes = 2
episode_rewards = []


for episode in range(n_episodes):
    obs = vec_env.reset()
    done = False
    total_reward = 0.0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        # Accumulate reward.
        if isinstance(reward, (list, np.ndarray)):
            total_reward += reward[0]
        else:
            total_reward += reward
        
        # vectorized environments 
        if isinstance(done, (list, np.ndarray)):
            done = all(done)
    
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")


avg_reward = np.mean(episode_rewards)
print(f"\nAverage Reward over {n_episodes} episodes: {avg_reward}")


episodes = np.arange(1, n_episodes + 1)
plt.figure(figsize=(10, 6))
plt.plot(episodes, episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Evaluation: Total Reward per Episode")
plt.grid(True)
plt.show()
