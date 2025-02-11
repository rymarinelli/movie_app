from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from deployment_env import DeploymentEnv
from tqdm import tqdm


class TqdmCallback(BaseCallback):
    """
    Callback for creating a progress to monitor training 
    """
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(self.locals.get("n_steps", 1))
        
        # If exceeded, return False to stop training.
        if self.num_timesteps >= self.total_timesteps:
            return False 
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


#Realistic Usage will run the reccomendation model
env = DeploymentEnv(base_url="http://127.0.0.1:5000", realistic_usage=True, wait_time=1)
vec_env = DummyVecEnv([lambda: env])


model = PPO("MlpPolicy", vec_env, verbose=1)

# Define total timesteps.
total_timesteps = 2

#progress bar callback.
tqdm_callback = TqdmCallback(total_timesteps=total_timesteps)


model.learn(total_timesteps=total_timesteps, callback=tqdm_callback)


model.save("deployment_scaling_agent")
print("Training complete. Model saved as deployment_scaling_agent.zip")
