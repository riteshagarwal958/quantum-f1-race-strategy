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

TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025]

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_f1_strategist_final")
RESULTS_DIR = "results"

RANDOM_EPISODES = 100
SEED = 0

ACTION_NAMES = {0: "STAY", 1: "PIT_SOFT", 2: "PIT_MEDIUM", 3: "PIT_HARD"}
COMPOUND_TO_ACTION = {0: 1, 1: 2, 2: 3}


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
    that haven't been generated yet."""
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


def evaluate_entries(model, entries, label):
    """Run random / real-strategy / DQN comparisons over every
    (track, year, driver, scenario) tuple in `entries`."""
    reward_rows, action_rows = [], []

    for track, year, driver, scenario in entries:
        print(f"\n{'=' * 10}\n{track.upper()} {year} ({label})\n{'=' * 10}")
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
            "year": year,
            "driver": driver,
            "random_reward": round(random_reward, 3),
            "real_strategy_reward": round(real_reward, 3),
            "dqn_reward": round(dqn_reward, 3),
        })
        action_rows.append({
            "track": track,
            "year": year,
            "driver": driver,
            "pits": dqn_pits,
            "finish_position": dqn_pos,
            **{ACTION_NAMES[a]: counts[a] for a in ACTION_NAMES},
        })

    return reward_rows, action_rows


def save_results(reward_rows, action_rows, results_csv, actions_csv, label):
    results = pd.DataFrame(
        reward_rows,
        columns=[
            "track",
            "year",
            "driver",
            "random_reward",
            "real_strategy_reward",
            "dqn_reward",
        ],
    )
    results.to_csv(results_csv, index=False)
    pd.DataFrame(action_rows).to_csv(actions_csv, index=False)

    print(f"\n{'=' * 10}\n{label} SUMMARY\n{'=' * 10}")
    print(results.to_string(index=False))
    print(f"Saved {results_csv} and {actions_csv}")


def main():
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise SystemExit(f"DQN model not found: {MODEL_PATH}.zip\n"
        "Run main_dqn.py first.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = DQN.load(MODEL_PATH)

    # --- Train years (2022-2024) ---
    train_entries = load_scenarios(TRAIN_YEARS, TRACKS, DRIVERS)
    if train_entries:
        train_reward_rows, train_action_rows = evaluate_entries(model, train_entries, "TRAIN")
        save_results(
            train_reward_rows,
            train_action_rows,
            os.path.join(RESULTS_DIR, "dqn_results_train.csv"),
            os.path.join(RESULTS_DIR, "dqn_action_analysis_train.csv"),
            "TRAIN",
        )
    else:
        print("No train scenarios found -- skipping train evaluation.")

    # --- Held-out test year (2025) ---
    test_entries = load_scenarios(TEST_YEARS, TRACKS, DRIVERS)
    if test_entries:
        test_reward_rows, test_action_rows = evaluate_entries(model, test_entries, "TEST")
        save_results(
            test_reward_rows,
            test_action_rows,
            os.path.join(RESULTS_DIR, "dqn_results_test.csv"),
            os.path.join(RESULTS_DIR, "dqn_action_analysis_test.csv"),
            "TEST",
        )
    else:
        print("No test scenarios found -- skipping test evaluation.")


if __name__ == "__main__":
    main()
