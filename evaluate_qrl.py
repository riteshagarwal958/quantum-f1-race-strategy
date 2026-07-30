

import csv
import random
import numpy as np
import torch

# Action indices, matching NUM_ACTIONS=4 / the pit_laps convention already used
# in final_evaluation() (action != 0 means "some pit stop happened").
ACTION_NAMES = {0: "STAY", 1: "PIT_SOFT", 2: "PIT_MEDIUM", 3: "PIT_HARD"}


def _split_track_year(name):
    """'Silverstone_2025' -> ('Silverstone', '2025'). Falls back to
    (name, '') if there's no trailing 4-digit year segment."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0], parts[1]
    return name, ""


def run_episode(env, action_fn):
    """Runs one full episode under action_fn(obs, env) -> action (int).
    (env is passed alongside obs so lap-indexed replay strategies, like
    real_strategy_action_fn, can read env.current_lap without having to
    reverse-engineer it from the normalized observation vector.)
    Returns (total_reward, action_counts dict, pit_laps list, final_info)."""
    obs, _ = env.reset()
    total_reward = 0.0
    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    pit_laps = []
    done = False
    info = {}

    while not done:
        action = action_fn(obs, env)
        action_counts[ACTION_NAMES[action]] += 1
        if action != 0:
            pit_laps.append(env.current_lap + 1)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward, action_counts, pit_laps, info


def random_action_fn(num_actions):
    """Uniform random action selector, matching gymnasium's convention of
    sampling any of the NUM_ACTIONS discrete actions with equal probability."""
    def fn(obs, env):
        return random.randint(0, num_actions - 1)
    return fn


def agent_action_fn(policy, obs_normalizer, device):
    """Greedy (argmax) action selector from a trained QRLActorCritic policy."""
    def fn(obs, env):
        probs = policy.get_action_probs(obs, obs_normalizer)
        return probs.argmax().item()
    return fn


# Inverse of F1StrategyEnv.ACTION_TO_COMPOUND = {1: 0, 2: 1, 3: 2}
COMPOUND_TO_ACTION = {0: 1, 1: 2, 2: 3}


def real_strategy_action_fn(scenario):
    """Replays the real, historical pit-stop strategy from `scenario`.

    Uses scenario.actual_compounds (dict: lap_number -> compound_code, built
    directly from FastF1's per-lap Compound column for every racing lap) and
    detects real pit events as compound *transitions* between consecutive
    laps, rather than trusting scenario.actual_pit_laps' exact lap-numbering
    convention directly. This is deliberate: FastF1's PitInTime/PitOutTime lap
    semantics (whether the pit-in lap's Compound value reflects the outgoing
    or incoming tyre) aren't unambiguous across data sources, whereas
    actual_compounds is a plain per-lap record that matches exactly how
    F1StrategyEnv itself tracks self.tyre_compound lap by lap -- so replaying
    off compound transitions guarantees self-consistency with the env.

    scenario.actual_pit_laps is used only as a sanity check: if its count
    disagrees with the number of compound transitions found, a warning is
    printed (worth investigating manually if it ever fires, but it doesn't
    block replay).
    """
    compounds = scenario.actual_compounds
    n_laps = scenario.n_laps

    pit_at_lap = {}  # lap_number -> new compound code to pit onto
    prev_compound = scenario.start_compound
    for lap in range(1, n_laps + 1):
        comp = compounds.get(lap, prev_compound)  # hold last known compound over any gap
        if comp != prev_compound:
            pit_at_lap[lap] = comp
        prev_compound = comp

    if len(pit_at_lap) != len(scenario.actual_pit_laps):
        print(f"[real_strategy_action_fn] WARNING ({scenario.track}): detected "
              f"{len(pit_at_lap)} compound transitions but scenario.actual_pit_laps "
              f"lists {len(scenario.actual_pit_laps)} laps -- lap-numbering convention "
              f"may differ by one somewhere; double-check this baseline's numbers.")

    def fn(obs, env):
        lap_number = env.current_lap + 1  # matches F1StrategyEnv.step()'s own `lap_number`
        if lap_number in pit_at_lap:
            return COMPOUND_TO_ACTION[pit_at_lap[lap_number]]
        return 0

    return fn


def evaluate_all_policies(policy, obs_normalizer, scenarios, device,
                           num_actions=4, real_strategy_fn=real_strategy_action_fn,
                           random_seed=None):
    """
    scenarios: dict of {name: scenario}, where name is either "<Track>_<Year>"
    (as produced by main.load_scenarios_for_years) or a plain track name.

    Returns two lists of dict rows, matching the two CSV tables:
        action_rows:  track, year, driver, pits, finish_position, STAY, PIT_SOFT, PIT_MEDIUM, PIT_HARD
        reward_rows:  track, year, driver, random_reward, real_strategy_reward, qrl_reward
    """
    if random_seed is not None:
        random.seed(random_seed)

    from f1_strategy_env import F1StrategyEnv  # local import, same as main.py

    action_rows = []
    reward_rows = []

    agent_fn = agent_action_fn(policy, obs_normalizer, device)
    rand_fn = random_action_fn(num_actions)

    for name, scenario in scenarios.items():
        track, year = _split_track_year(name)
        driver = scenario.driver

        # --- Random policy ---
        env = F1StrategyEnv(scenario=scenario)
        random_reward, _, _, _ = run_episode(env, rand_fn)

        # --- Trained agent (QRL) ---
        env = F1StrategyEnv(scenario=scenario)
        agent_reward, agent_counts, agent_pit_laps, agent_info = run_episode(env, agent_fn)

        # --- Real historical strategy (needs real_strategy_fn implemented) ---
        if real_strategy_fn is not None:
            env = F1StrategyEnv(scenario=scenario)
            real_fn = real_strategy_fn(scenario)
            real_reward, real_counts, real_pit_laps, real_info = run_episode(env, real_fn)
        else:
            real_reward = None
            real_counts = {name: None for name in ACTION_NAMES.values()}
            real_info = {"pits_done": None, "position": None}

        action_rows.append({
            "track": track,
            "year": year,
            "driver": driver,
            "pits": real_info.get("pits_done"),
            "finish_position": real_info.get("position"),
            "STAY": real_counts["STAY"],
            "PIT_SOFT": real_counts["PIT_SOFT"],
            "PIT_MEDIUM": real_counts["PIT_MEDIUM"],
            "PIT_HARD": real_counts["PIT_HARD"],
        })

        reward_rows.append({
            "track": track,
            "year": year,
            "driver": driver,
            "random_reward": round(random_reward, 3),
            "real_strategy_reward": round(real_reward, 3) if real_reward is not None else None,
            "qrl_reward": round(agent_reward, 3),
        })

        label = f"{track} {year}".strip()
        print(f"{label:<18} random={random_reward:>10.2f}  "
              f"real_strategy={'N/A' if real_reward is None else f'{real_reward:.2f}':>10}  "
              f"qrl={agent_reward:>8.2f}  pits={agent_info.get('pits_done')} "
              f"P{agent_info.get('position')}")

    return action_rows, reward_rows


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {path}")


def run_baseline_comparison(policy, obs_normalizer, scenarios, device,
                             out_dir="qrl_logs", real_strategy_fn=real_strategy_action_fn):
    """Convenience wrapper: evaluate + save both CSVs, same layout as the
    tables you already generated for the DQN baseline.

    out_dir is caller-controlled -- pass a distinct path (e.g.
    "qrl_logs/test_2025") when evaluating a held-out test year so it doesn't
    overwrite training-set comparison CSVs."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    action_rows, reward_rows = evaluate_all_policies(
        policy, obs_normalizer, scenarios, device, real_strategy_fn=real_strategy_fn
    )

    save_csv(action_rows, f"{out_dir}/qrl_action_breakdown.csv",
              fieldnames=["track", "year", "driver", "pits", "finish_position",
                          "STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"])
    save_csv(reward_rows, f"{out_dir}/qrl_reward_comparison.csv",
              fieldnames=["track", "year", "driver", "random_reward",
                          "real_strategy_reward", "qrl_reward"])

    return action_rows, reward_rows
