import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

STATE_FEATURES = [
    "remaining_laps_norm",
    "compound_soft",
    "compound_medium",
    "compound_hard",
    "tyre_age_norm",
    "lap_delta_norm",
    "position_norm",
    "gap_ahead_norm",
    "gap_behind_norm",
    "gap_to_leader_norm",
    "pits_done_norm",
    "safety_car_active",
    "deg_penalty_norm",
]

@dataclass
class RaceScenario:
    track: str
    driver: str
    n_laps: int
    n_drivers: int

    # Pace / tyre model (seconds).
    median_pace: float                 # (pace baseline)
    pit_loss: float
    compound_base_pace: dict           # base lap time on fresh tyres
    compound_deg_rate: dict            # seconds added per lap of age

    # Race context.
    start_compound: int
    safety_car_laps: set
    field_cumulative: pd.DataFrame     # index=lap, cols=driver, values=cumulative seconds
    field_sorted: dict                 # lap -> sorted np.ndarray of COMPETITOR times (fast gaps)

    # Real team strategy (for the real-strategy baseline + replay/benchmark).
    actual_pit_laps: list
    actual_compounds: dict             # lap_number -> compound_code
    actual_positions: dict             # lap_number -> position
    cliff_age: int                     # longest real stint -> track-specific degradation cliff

class F1DataPipeline:
    # Normalization constants
    MAX_TYRE_LIFE = 40.0               # normalizes tyre age
    MAX_STOPS = 3                      # normalizes pits_done
    MAX_DEG_PENALTY = 5.0              # normalizes deg_penalty
    GAP_SCALE = 60.0                   # normalizes gap_ahead / gap_behind

    # Fallbacks when the data is too sparse to fit
    DEFAULT_DEG_RATE = {0: 0.08, 1: 0.05, 2: 0.03}   # SOFT / MEDIUM / HARD (s/lap)
    DEFAULT_PIT_LOSS = {
        'Silverstone': 23.5, 'Monza': 19.1, 'Monaco': 25.0,
        'Spa': 22.0, 'Suzuka': 23.0, 'Singapore': 29.0,
    }

    def __init__(self, cache_dir='f1_cache'):
        import fastf1
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)
        self.compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

    def _load_session_laps(self, year, track, driver):
        import fastf1
        session = fastf1.get_session(year, track, 'R')
        session.load(telemetry=False, weather=False)
        all_laps = session.laps
        driver_laps = all_laps.pick_driver(driver).copy()
        if driver_laps.empty:
            raise ValueError(f"No lap data found for {driver} at {track} {year}")
        return all_laps, driver_laps

    def fetch_race_scenario(self, year, track, driver):
        """Return a RaceScenario with simulator parameters fitted from real data."""
        all_laps, driver_laps = self._load_session_laps(year, track, driver)

        compound_raw = driver_laps['Compound'].map(self.compound_map).fillna(1).astype(int)
        lap_times = driver_laps['LapTime'].dt.total_seconds()
        is_normal = driver_laps['PitOutTime'].isna() & lap_times.notna()

        median_pace = lap_times[is_normal].median()
        if pd.isna(median_pace):
            median_pace = 90.0
        median_pace = float(median_pace)

        rolling_pace = (
            lap_times[is_normal].rolling(7, min_periods=1).median()
            .reindex(driver_laps.index).ffill().bfill().fillna(median_pace)
        )

        base_pace, deg_rate = self._fit_tyre_models(
            driver_laps, lap_times, is_normal, compound_raw, median_pace
        )
        pit_loss = self._measure_pit_loss(driver_laps, lap_times, rolling_pace, track)

        sc_mask = driver_laps['TrackStatus'].astype(str).str.contains(r'[4567]', regex=True)
        safety_car_laps = set(driver_laps.loc[sc_mask, 'LapNumber'].astype(int).tolist())

        field_cumulative = self._build_field_reference(all_laps)
        n_drivers = max(field_cumulative.shape[1], 1)
        field_sorted = self._build_field_sorted(field_cumulative, driver)

        lap_index = driver_laps['LapNumber'].astype(int)
        pit_in_mask = driver_laps['PitInTime'].notna()
        actual_pit_laps = driver_laps.loc[pit_in_mask, 'LapNumber'].astype(int).tolist()

        # Compute track-specific degradation cliff from real stint lengths.
        stint_lengths = []
        last_pit = 1
        for pit in actual_pit_laps:
            stint_lengths.append(pit - last_pit)
            last_pit = pit
        stint_lengths.append(int(driver_laps['LapNumber'].max()) - last_pit + 1)
        cliff_age = int(np.clip(max(stint_lengths), 20, 40)) if stint_lengths else 25

        actual_compounds = dict(zip(lap_index, compound_raw.astype(int)))
        if 'Position' in driver_laps.columns:
            positions = driver_laps['Position'].ffill().bfill()
            actual_positions = dict(zip(lap_index, positions.fillna(0).astype(int)))
        else:
            actual_positions = {}

        start_compound = int(compound_raw.iloc[0]) if len(compound_raw) else 0

        return RaceScenario(
            track=track,
            driver=driver,
            n_laps=int(driver_laps['LapNumber'].max()),
            n_drivers=int(n_drivers),
            median_pace=median_pace,
            pit_loss=float(pit_loss),
            compound_base_pace=base_pace,
            compound_deg_rate=deg_rate,
            start_compound=start_compound,
            safety_car_laps=safety_car_laps,
            field_cumulative=field_cumulative,
            field_sorted=field_sorted,
            actual_pit_laps=actual_pit_laps,
            actual_compounds=actual_compounds,
            actual_positions=actual_positions,
            cliff_age=cliff_age,
        )

    def _measure_pit_loss(self, driver_laps, lap_times, rolling_pace, track):
        """Estimate time lost to a pit stop (s), with a per-track fallback."""
        try:
            pit_laps = driver_laps[driver_laps['PitOutTime'].notna()]
            pit_losses = []
            for idx in pit_laps.index:
                pit_lap_time = lap_times.loc[idx]
                baseline_pace = rolling_pace.loc[idx]
                if pd.notna(pit_lap_time) and pd.notna(baseline_pace):
                    pit_losses.append(max(0.0, pit_lap_time - baseline_pace))
            if len(pit_losses) > 0:
                return float(np.mean(pit_losses))
            raise ValueError("No valid pit losses")
        except (AttributeError, KeyError, ValueError, IndexError):
            return float(self.DEFAULT_PIT_LOSS.get(track, 24.0))

    def _fit_tyre_models(self, driver_laps, lap_times, is_normal, compound_raw, median_pace):
        ages = driver_laps['TyreLife'].fillna(1.0)
        base_pace, deg_rate = {}, {}

        low, high = 0.97 * median_pace, 1.10 * median_pace  # plausible base-pace band
        min_laps, max_deg = 5, 0.20

        for code in sorted(set(int(c) for c in compound_raw.unique())):
            mask = is_normal & (compound_raw == code)
            x = ages[mask].to_numpy(dtype=float)
            y = lap_times[mask].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]

            fitted = None
            if len(x) >= min_laps and np.ptp(x) >= 3:
                slope, intercept = np.polyfit(x, y, 1)
                slope = float(np.clip(slope, 0.0, max_deg))
                if low <= intercept <= high:
                    fitted = (float(intercept), slope)

            if fitted is None:
                base = float(np.median(y)) if len(y) else median_pace
                if not (low <= base <= high):
                    base = median_pace
                fitted = (base, self.DEFAULT_DEG_RATE.get(code, 0.05))

            base_pace[code], deg_rate[code] = fitted

        for code in self.compound_map.values():
            base_pace.setdefault(code, median_pace)
            deg_rate.setdefault(code, self.DEFAULT_DEG_RATE.get(code, 0.05))

        return base_pace, deg_rate

    def _build_field_reference(self, all_laps):
        """Cumulative race time (s) per driver per lap: index=lap, cols=driver."""
        times = all_laps.groupby(['LapNumber', 'Driver'])['Time'].min().unstack('Driver')
        field = times.apply(lambda s: s.dt.total_seconds())
        field = field.ffill().bfill()
        field.index = field.index.astype(int)
        return field

    def _build_field_sorted(self, field_cumulative, driver):
        """Per-lap ascending competitor times (agent excluded) for O(log n) gaps."""
        competitors = field_cumulative.drop(columns=[driver], errors='ignore')
        return {
            int(lap): np.sort(competitors.loc[lap].dropna().to_numpy(dtype=float))
            for lap in competitors.index
        }
