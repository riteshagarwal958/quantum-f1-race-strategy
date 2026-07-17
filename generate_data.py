import os
import pickle

from data_fetch import F1DataPipeline

TRACKS = [
    "Silverstone",
    "Monza",
    "Monaco",
]
DRIVERS = [
    "HAM",
]
YEAR = 2023
OUT_DIR = "data"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pipeline = F1DataPipeline()

    for driver in DRIVERS:
        for track in TRACKS:

            print(f"Generating {track} - {driver}")

            scenario = pipeline.fetch_race_scenario(
                YEAR,
                track,
                driver,
            )

            path = os.path.join(
                OUT_DIR,
                f"{track.lower()}_{YEAR}_{driver.lower()}.pkl"
            )

            with open(path, "wb") as f:
                pickle.dump(scenario, f)

            print(
                f"Saved {path} "
                f"({scenario.n_laps} laps, cliff_age={scenario.cliff_age})"
            )

if __name__ == "__main__":
    main()
