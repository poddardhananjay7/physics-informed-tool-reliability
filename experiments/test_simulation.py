from src.simulation.degradation_simulation import simulate_degradation

if __name__ == "__main__":
    life, df = simulate_degradation(damping=2, force_amp=100)
    print("Lifetime:", life)
    print(df.head())
