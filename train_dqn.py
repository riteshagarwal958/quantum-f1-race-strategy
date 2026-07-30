import os
import pickle

import gymnasium as gym
import pandas as pd
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

# Curriculum: train across all (track, year) combinations in TRAIN_YEARS.
TRAIN_YEARS = [2022, 2023, 2024]

# Held-out year(s), evaluated only after training -- never trained on.
TEST_YEARS = [2025]

DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "tensorboard_logs"
RESULTS_DIR = "results"

TOTAL_TIMESTEPS = 600_000
SEED = 42
N_EVAL_EPISODES = 10


def load_scenario(track, year, driver):
    path = os.path.join(
        DATA_DIR,
        f"{track.lower()}_{year}_{driver.lower()}.pkl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_scenarios(years, tracks, drivers):
    """Load every (track, year, driver) scenario available, skipping any
    that haven't been generated yet. Returns a list of
    (track, year, driver, scenario) tuples."""
    entries = []
    for driver in drivers:
        for year in years:
            for track in tracks:
                try:
                    scenario = load_scenario(track, year, driver)
                except FileNotFoundError:
                    print(f"  [SKIP] missing scenario: {track} {year} {driver}")
                    continue
                entries.append((track, year, driver, scenario))
    return entries


class MultiTrackEnv(gym.Env):
    """Each reset picks a random (track, year) scenario -> no catastrophic
    forgetting across the curriculum."""
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


def evaluate_and_save(model, entries, csv_path, label):
    """Run evaluate_policy on every (track, year, driver, scenario) in
    `entries` and save a tidy results table to `csv_path`."""
    rows = []
    print(f"\n=== {label} evaluation ===")
    for track, year, driver, scenario in entries:
        mean_reward, std_reward = evaluate_policy(
            model,
            F1StrategyEnv(scenario=scenario),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
        )
        print(f"{track:12} {year} {driver:4} : {mean_reward:8.2f} +/- {std_reward:5.2f}")
        rows.append({
            "track": track,
            "year": year,
            "driver": driver,
            "mean_reward": round(mean_reward, 3),
            "std_reward": round(std_reward, 3),
            "n_eval_episodes": N_EVAL_EPISODES,
        })

    df = pd.DataFrame(
        rows,
        columns=["track", "year", "driver", "mean_reward", "std_reward", "n_eval_episodes"],
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Loading TRAIN scenarios: years={TRAIN_YEARS}, tracks={TRACKS} ===")
    train_entries = load_scenarios(TRAIN_YEARS, TRACKS, DRIVERS)
    if not train_entries:
        raise SystemExit(
            "No training scenarios found in 'data/'. Run generate_data.py first."
        )

    train_scenarios = [scenario for (_, _, _, scenario) in train_entries]
    train_env = MultiTrackEnv(train_scenarios)

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

    # --- Evaluation: train years and held-out test year, saved separately ---
    evaluate_and_save(
        model,
        train_entries,
        os.path.join(RESULTS_DIR, "dqn_train_results.csv"),
        f"TRAIN ({', '.join(map(str, TRAIN_YEARS))})",
    )

    print(f"\n=== Loading TEST scenarios: years={TEST_YEARS}, tracks={TRACKS} ===")
    test_entries = load_scenarios(TEST_YEARS, TRACKS, DRIVERS)
    if test_entries:
        evaluate_and_save(
            model,
            test_entries,
            os.path.join(RESULTS_DIR, "dqn_test_results.csv"),
            f"TEST ({', '.join(map(str, TEST_YEARS))}, held-out)",
        )
    else:
        print("No test scenarios found -- skipping test evaluation.")


if __name__ == "__main__":
    main()
