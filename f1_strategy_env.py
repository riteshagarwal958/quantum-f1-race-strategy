import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from data_fetch import STATE_FEATURES, F1DataPipeline as _DL

MAX_TYRE_LIFE = _DL.MAX_TYRE_LIFE
MAX_STOPS = _DL.MAX_STOPS
MAX_DEG_PENALTY = _DL.MAX_DEG_PENALTY
GAP_SCALE = _DL.GAP_SCALE


class F1StrategyEnv(gym.Env):
    metadata = {"render_modes": ["console", "human"]}
    ACTION_TO_COMPOUND = {1: 0, 2: 1, 3: 2}

    # Reward shaping.
    PACE_SCALE = 0.10
    POSITION_SCALE = 0.05
    FINISH_SCALE = 1.00
    EXTRA_STOP_PENALTY = 6.0
    MIN_STINT_LAPS = 8
    SHORT_STINT_PENALTY = 4.0
    SC_PIT_FACTOR = 0.6  
    SC_PACE_FACTOR = 1.30

    # Tyre-degradation cliff (overridden per scenario from cliff_age).
    CLIFF_AGE = 25
    CLIFF_RATE = 0.03

    def __init__(self, data_path: str = "data/race_scenario.pkl", scenario=None):
        super().__init__()

        self.scenario = scenario if scenario is not None else pd.read_pickle(data_path)
        s = self.scenario

        self.warmup_penalty = {
            0: 1.34,
            1: 1.60,
            2: 2.95,
        }

        self.n_laps = int(s.n_laps)
        if self.n_laps <= 0:
            raise ValueError("scenario has no laps")

        self.n_drivers = int(s.n_drivers)
        self.median_pace = float(s.median_pace)
        self.pit_loss = float(s.pit_loss)
        self.base_pace = dict(s.compound_base_pace)
        self.deg_rate = dict(s.compound_deg_rate)
        self.start_compound = int(s.start_compound)
        self.safety_car_laps = set(s.safety_car_laps)

        # Track-specific cliff from real stint lengths.
        self.CLIFF_AGE = int(getattr(s, "cliff_age", 25))

        actual_positions = getattr(s, "actual_positions", None) or {}
        self.start_position = int(actual_positions.get(1, self.n_drivers))

        self.field_sorted = s.field_sorted
        self._fs_laps = np.array(sorted(self.field_sorted)) if self.field_sorted else np.array([])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(STATE_FEATURES),), dtype=np.float32
        )

        self._init_state()

    def _init_state(self):
        self.pace_loss = 0.0
        self.current_lap = 0
        self.total_time = 0.0
        self.tyre_compound = self.start_compound
        self.tyre_age = 0
        self.pits_done = 0
        self.position = self.start_position
        self.gap_ahead = 0.0
        self.gap_behind = 0.0
        self.lap_delta = 0.5
        self.gap_to_leader_norm = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        lap_number = self.current_lap + 1
        under_sc = lap_number in self.safety_car_laps

        # Pit consequence
        pit_cost = 0.0
        extra_stop_penalty = 0.0
        stint_len = None
        
        if action in self.ACTION_TO_COMPOUND:
            stint_len = self.tyre_age
            self.tyre_compound = self.ACTION_TO_COMPOUND[action]
            self.tyre_age = 0
            self.pits_done += 1
            pit_cost = self.pit_loss * (self.SC_PIT_FACTOR if under_sc else 1.0)
            self.total_time += pit_cost
            if self.pits_done > 1:
                extra_stop_penalty = (self.pits_done - 1) * self.EXTRA_STOP_PENALTY

        # --- Drive the lap (warmup + degradation + cliff) ---

        self.pace_loss = self._pace_loss(self.tyre_compound, self.tyre_age)
        lap_time = self.base_pace[self.tyre_compound] + self.pace_loss

        if under_sc:
            lap_time *= self.SC_PACE_FACTOR

        if self.tyre_age == 0:
            warmup = self.warmup_penalty[self.tyre_compound]

            if under_sc:
                warmup *= 0.5

            lap_time += warmup

        self.total_time += lap_time
        self.tyre_age += 1
        self.current_lap += 1

        # --- Computed obs signals ---
        self.lap_delta = float(np.clip((lap_time - self.median_pace + 5.0) / 10.0, 0.0, 1.0))

        # Inline positions and gaps calculation
        comp = self._competitor_times(self.current_lap)
        if comp.size == 0:
            self.position, self.gap_ahead, self.gap_behind = 1, 0.0, 0.0
            leader_time = self.total_time
        else:
            idx = int(np.searchsorted(comp, self.total_time))
            self.position = idx + 1
            gap_ahead_raw = self.total_time - comp[idx - 1] if idx > 0 else 0.0
            gap_behind_raw = comp[idx] - self.total_time if idx < comp.size else 0.0
            self.gap_ahead = min(max(0.0, float(gap_ahead_raw)), GAP_SCALE,)
            self.gap_behind = min(max(0.0, float(gap_behind_raw)), GAP_SCALE,)
            leader_time = float(comp[0])

        gap_to_leader = max(0.0, self.total_time - leader_time)
        self.gap_to_leader_norm = float(min(gap_to_leader, GAP_SCALE) / GAP_SCALE)

        terminated = self.current_lap >= self.n_laps
        truncated = False
        
        # Inline reward calculation
        lap_cost = lap_time + pit_cost

        if under_sc:
            pace_reward = 0.0
        else:
            pace_reward = (self.median_pace - lap_cost) * self.PACE_SCALE

        reward = pace_reward
        threshold = 3 if under_sc else self.MIN_STINT_LAPS
        
        if stint_len is not None and stint_len < threshold:
            reward -= (threshold - stint_len) * self.SHORT_STINT_PENALTY
            
        reward += ((self.n_drivers - self.position) / max(self.n_drivers - 1, 1)) * self.POSITION_SCALE
        
        if terminated:
            reward += (self.n_drivers - self.position) * self.FINISH_SCALE
            
        reward -= extra_stop_penalty

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # --- Retained Helper Methods ---

    def _pace_loss(self, compound, age):
        loss = age * self.deg_rate.get(compound, 0.0)
        if age > self.CLIFF_AGE:
            over = age - self.CLIFF_AGE
            loss += self.CLIFF_RATE * (over ** 1.5)
        return loss

    def _competitor_times(self, lap):
        if self._fs_laps.size == 0:
            return np.array([])
        if lap in self.field_sorted:
            return self.field_sorted[lap]
        i = int(np.searchsorted(self._fs_laps, lap, side="right")) - 1
        i = max(i, 0)
        return self.field_sorted[int(self._fs_laps[i])]

    def _get_obs(self):
        c = self.tyre_compound
        deg_penalty = self.pace_loss
        lap_number = min(self.current_lap + 1, self.n_laps)

        obs = np.array([
            max(0.0, 1.0 - self.current_lap / self.n_laps),               # remaining_laps_norm
            1.0 if c == 0 else 0.0,                                       # compound_soft
            1.0 if c == 1 else 0.0,                                       # compound_medium
            1.0 if c == 2 else 0.0,                                       # compound_hard
            min(self.tyre_age, MAX_TYRE_LIFE) / MAX_TYRE_LIFE,            # tyre_age_norm
            self.lap_delta,                                               # lap_delta_norm
            (self.n_drivers - self.position) / max(self.n_drivers - 1, 1),# position_norm P1=1, P20=0
            min(self.gap_ahead, GAP_SCALE) / GAP_SCALE,                   # gap_ahead_norm
            min(self.gap_behind, GAP_SCALE) / GAP_SCALE,                  # gap_behind_norm
            self.gap_to_leader_norm,                                      # gap_to_leader_norm
            min(self.pits_done, MAX_STOPS) / MAX_STOPS,                   # pits_done_norm
            1.0 if lap_number in self.safety_car_laps else 0.0,           # safety_car_active
            min(deg_penalty, MAX_DEG_PENALTY) / MAX_DEG_PENALTY,          # deg_penalty_norm
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        return {
            "lap": self.current_lap,
            "position": self.position,
            "total_time": self.total_time,
            "pits_done": self.pits_done,
            "tyre_compound": self.tyre_compound,
            "tyre_age": self.tyre_age,
            "gap_ahead": self.gap_ahead,
            "gap_behind": self.gap_behind,
        }

    def render(self, mode="console"):
        if mode in ("console", "human"):
            print("=" * 50)
            print(f"Lap {self.current_lap}/{self.n_laps}")
            print("=" * 50)
            print(f"  Compound: {self.tyre_compound} | Tyre Age: {self.tyre_age}")
            print(f"  Position: P{self.position}/{self.n_drivers}")
            print(f"  Gap ahead: {self.gap_ahead:.1f}s | Gap behind: {self.gap_behind:.1f}s")
            print(f"  Pit Stops: {self.pits_done}")
            print(f"  Total Time: {self.total_time:.1f}s")
