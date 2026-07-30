import os
import pickle

from data_loader import F1DataPipeline

TRACKS = ["Silverstone", "Monza", "Monaco"]
DRIVER = "HAM"

# Years used to build the training set (curriculum across tracks x years).
TRAIN_YEARS = [2022, 2023, 2024]

# Held-out year(s) used only for final evaluation / testing, never trained on.
TEST_YEARS = [2025]

OUT_DIR = "data"


def generate_for_year(pipeline, year):
    """Fetch + pickle a scenario for every track for a single year.
    Returns the list of (track, path) pairs successfully saved, and a list
    of (track, error) pairs that failed (e.g. session not available yet)."""
    saved = []
    failed = []
    for track in TRACKS:
        try:
            scenario = pipeline.fetch_race_scenario(year, track, DRIVER)
        except Exception as e:
            print(f"  [SKIP] {track} {year}: {e}")
            failed.append((track, str(e)))
            continue

        path = os.path.join(OUT_DIR, f"{track.lower()}_{year}_{DRIVER.lower()}.pkl")
        with open(path, "wb") as f:
            pickle.dump(scenario, f)
        print(f"Saved {path}  ({scenario.n_laps} laps, cliff_age={scenario.cliff_age})")
        saved.append((track, path))
    return saved, failed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pipeline = F1DataPipeline()

    print(f"=== Generating TRAIN data: years={TRAIN_YEARS}, tracks={TRACKS} ===")
    all_failed = []
    for year in TRAIN_YEARS:
        print(f"\n--- {year} (train) ---")
        _, failed = generate_for_year(pipeline, year)
        all_failed.extend((t, year, e) for t, e in failed)

    print(f"\n=== Generating TEST data: years={TEST_YEARS}, tracks={TRACKS} ===")
    for year in TEST_YEARS:
        print(f"\n--- {year} (test) ---")
        _, failed = generate_for_year(pipeline, year)
        all_failed.extend((t, year, e) for t, e in failed)

    if all_failed:
        print("\n=== Some scenarios could not be generated ===")
        for track, year, err in all_failed:
            print(f"  {track} {year}: {err}")
    else:
        print("\nAll scenarios generated successfully.")


if __name__ == "__main__":
    main()
