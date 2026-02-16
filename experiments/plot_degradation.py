import matplotlib.pyplot as plt
from src.simulation.degradation_simulation import simulate_degradation


def plot_degradation():

    lifetime, df = simulate_degradation(damping=2, force_amp=100)

    plt.figure()
    plt.plot(df["cycle"], df["damage"], label="True Damage")
    plt.plot(df["cycle"], df["proxy_damage"], label="Proxy Damage")
    plt.xlabel("Cycle")
    plt.ylabel("Damage")
    plt.title("Damage Accumulation vs Cycle")
    plt.legend()
    plt.savefig("docs/degradation_curve.png")
    plt.close()

    print("Degradation curve saved to docs/degradation_curve.png")


if __name__ == "__main__":
    plot_degradation()
