import pandas as pd

df = pd.read_csv("data/rul_dataset.csv")

print("Total rows:", len(df))

print("\n--- RUL Statistics ---")
print("Min RUL:", df["RUL"].min())
print("Max RUL:", df["RUL"].max())

print("\n--- RMS Statistics ---")
print("Min RMS:", df["rms"].min())
print("Max RMS:", df["rms"].max())

print("\n--- Peak Statistics ---")
print("Min Peak:", df["peak"].min())
print("Max Peak:", df["peak"].max())
