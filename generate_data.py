import os
import pickle

from data_loader import F1DataPipeline

TRACKS = [
    "Silverstone",
    "Monza",
    "Monaco",
]
DRIVERS = [
    "HAM",
]
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
OUT_DIR = "data"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pipeline = F1DataPipeline()

    for year in TRAIN_YEARS + [TEST_YEAR]:
        for driver in DRIVERS:
            for track in TRACKS:

                print(f"Generating {track} {year} - {driver}")

                scenario = pipeline.fetch_race_scenario(
                    year,
                    track,
                    driver,
                )

                path = os.path.join(
                    OUT_DIR,
                    f"{track.lower()}_{year}_{driver.lower()}.pkl"
                )

                with open(path, "wb") as f:
                    pickle.dump(scenario, f)

                print(
                    f"Saved {path} "
                    f"({scenario.n_laps} laps, cliff_age={scenario.cliff_age})"
                )

if __name__ == "__main__":
    main()
