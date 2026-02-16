import numpy as np
import pandas as pd
import time
from src.simulation.degradation_simulation import simulate_degradation


def generate_rul_dataset(damping_values, force_values):

    all_runs = []

    total_runs = len(damping_values) * len(force_values)
    run_counter = 0

    start_time = time.time()

    for damping in damping_values:
        for force in force_values:

            run_counter += 1
            print(f"\n=== Run {run_counter}/{total_runs} ===")
            print(f"Damping={damping}, Force={force}")

            run_start = time.time()

            lifetime, df = simulate_degradation(
                damping=damping,
                force_amp=force
            )

            run_time = time.time() - run_start

            print(f"Lifetime: {lifetime}")
            print(f"Run time: {run_time:.2f} seconds")

            all_runs.append(df)

            elapsed = time.time() - start_time
            avg_time = elapsed / run_counter
            remaining = avg_time * (total_runs - run_counter)

            print(f"Progress: {100 * run_counter / total_runs:.1f}%")
            print(f"Estimated remaining time: {remaining / 60:.2f} minutes")

    full_df = pd.concat(all_runs, ignore_index=True)

    full_df.to_csv("data/rul_dataset.csv", index=False)

    print("\nDataset generation complete.")
    print(f"Total rows: {len(full_df)}")

    return full_df
