import os
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

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
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_f1_strategist_final")

RESULTS_CSV = "dqn_results.csv"
ACTIONS_CSV = "dqn_action_analysis.csv"
RANDOM_EPISODES = 100
SEED = 0

ACTION_NAMES = {0: "STAY", 1: "PIT_SOFT", 2: "PIT_MEDIUM", 3: "PIT_HARD"}
COMPOUND_TO_ACTION = {0: 1, 1: 2, 2: 3}

def load_scenario(track, driver):
    path = os.path.join(
        DATA_DIR,
        f"{track.lower()}_{YEAR}_{driver.lower()}.pkl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)

def run_dqn_episode(model, env):
    obs, _ = env.reset()

    total = 0.0
    done = False
    info = {}

    counts = {a: 0 for a in ACTION_NAMES}
    pit_laps = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        counts[action] += 1

        if action != 0:
            pit_laps.append(env.current_lap + 1)

        obs, reward, terminated, truncated, info = env.step(action)

        total += reward
        done = terminated or truncated

    return total, counts, info["pits_done"], info["position"], pit_laps

def run_real_strategy(scenario, env):
    pit_laps = set(scenario.actual_pit_laps)
    compounds = scenario.actual_compounds
    obs, _ = env.reset()
    total, done, info = 0.0, False, {}
    while not done:
        lap = env.current_lap + 1
        action = COMPOUND_TO_ACTION.get(int(compounds.get(lap + 1, compounds.get(lap, 1))), 2) if lap in pit_laps else 0
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    return total, info["pits_done"], info["position"]

def run_random(env, n_episodes=RANDOM_EPISODES, seed=SEED):
    env.action_space.seed(seed)
    totals = []
    for _ in range(n_episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))

def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise SystemExit(f"DQN model not found: {MODEL_PATH}.zip\n"
        "Run train_dqn.py first.")
    model = DQN.load(MODEL_PATH)

    reward_rows, action_rows = [], []

    for driver in DRIVERS:
        for track in TRACKS:
            print(f"\n{'=' * 10}"
            f"\n{track.upper()}"
            f"\n{'=' * 10}")
            scenario = load_scenario(track, driver)

            print(f"Cliff Age       : {scenario.cliff_age} laps")
            print(f"Safety Car Laps : {sorted(scenario.safety_car_laps)}")

            random_reward = run_random(F1StrategyEnv(scenario=scenario))
            real_reward, real_pits, real_pos = run_real_strategy(
                scenario, F1StrategyEnv(scenario=scenario)
            )
            dqn_reward, counts, dqn_pits, dqn_pos, pit_laps = run_dqn_episode(
                model, F1StrategyEnv(scenario=scenario)
            )

            print(f"  Random        : {random_reward:+8.2f}")
            print(f"  Real strategy : {real_reward:+8.2f}  (pits={real_pits}, finish=P{real_pos})")
            print(f"  DQN agent     : {dqn_reward:+8.2f}  (pits={dqn_pits}, finish=P{dqn_pos})")
            print(f"  DQN pit laps  : {pit_laps}")
            print("  DQN actions   : " + ", ".join(f"{ACTION_NAMES[a]}={counts[a]}" for a in counts))

            reward_rows.append({
                "track": track,
                "driver": driver,
                "random_reward": round(random_reward, 3),
                "real_strategy_reward": round(real_reward, 3),
                "dqn_reward": round(dqn_reward, 3),
            })
            action_rows.append({
                "track": track,
                "driver": driver,
                "pits": dqn_pits,
                "finish_position": dqn_pos,
                **{ACTION_NAMES[a]: counts[a] for a in ACTION_NAMES},
            })

    results = pd.DataFrame(
        reward_rows,
        columns=[
            "track",
            "driver",
            "random_reward",
            "real_strategy_reward",
            "dqn_reward",
        ],
    )
    results.to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(action_rows).to_csv(ACTIONS_CSV, index=False)

    print(
        f"\n{'=' * 10}"
        f"\n{track.upper()} - {driver}"
        f"\n{'=' * 10}"
    )
    print(results.to_string(index=False))
    print(f"\nSaved {RESULTS_CSV} and {ACTIONS_CSV}")

if __name__ == "__main__":
    main()
