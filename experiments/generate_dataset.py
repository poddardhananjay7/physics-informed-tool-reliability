import numpy as np
from src.predictive.generate_rul_dataset import generate_rul_dataset

if __name__ == "__main__":

    damping_values = np.linspace(2, 10, 4)
    force_values = [80, 100]

    df = generate_rul_dataset(damping_values, force_values)

    print("Total rows:", len(df))
