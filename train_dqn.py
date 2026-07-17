import os
import pickle

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from f1_strategy_env import F1StrategyEnv

TRACKS = [
    "Silverstone",
    "Monza",
    "Monaco",
]

DRIVERS = [
    "HAM",
]

YEAR = 2023


DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "tensorboard_logs"

TOTAL_TIMESTEPS = 600_000
SEED = 42

def load_scenario(track, driver):
    path = os.path.join(
        DATA_DIR,
        f"{track.lower()}_{YEAR}_{driver.lower()}.pkl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)

class MultiTrackEnv(gym.Env):
    """Each reset picks a random track -> no forgetting."""
    metadata = {"render_modes": ["console", "human"]}

    def __init__(self, scenarios):
        super().__init__()
        self.envs = [F1StrategyEnv(scenario=s) for s in scenarios]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.env = self.envs[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        idx = self.np_random.integers(len(self.envs))
        self.env = self.envs[idx]

        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)
    
    def close(self):
        for env in self.envs:
            env.close()

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    scenario_list = []

    for driver in DRIVERS:
        for track in TRACKS:
            scenario_list.append(load_scenario(track, driver))

    train_env = MultiTrackEnv(scenario_list)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        seed=SEED,
        tensorboard_log=LOG_DIR,
    )
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="DQN_F1", progress_bar=True)
    finally:
        train_env.close()

    model_path = os.path.join(MODEL_DIR, "dqn_f1_strategist_final")
    model.save(model_path)
    print(f"\nSaved DQN baseline -> {model_path}.zip")

    print("\nFinal Evaluation")

    for driver in DRIVERS:
        print(f"\nDriver : {driver}")

        for track in TRACKS:

            scenario = load_scenario(track, driver)

            mean_reward, _ = evaluate_policy(
                model,
                F1StrategyEnv(scenario=scenario),
                n_eval_episodes=10,
                deterministic=True,
            )

            print(f"{track:12} : {mean_reward:8.2f}")

if __name__ == "__main__":
    main()
