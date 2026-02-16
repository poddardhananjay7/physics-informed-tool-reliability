import numpy as np
import pandas as pd
from src.dynamics.blade_dynamics import BladeDynamics
from src.fatigue.fatigue_model import FatigueModel
from src.wear.wear_model import WearModel
from src.monitoring.feature_extraction import MonitoringFeatures


def simulate_degradation(damping, force_amp,
                         stress_scale=2500,
                         fatigue_A=5e11,
                         fatigue_b=3,
                         max_cycles=200000):

    blade = BladeDynamics(
        mass=0.2,
        damping=damping,
        stiffness=10000,
        force_amp=force_amp,
        omega=200
    )

    fatigue = FatigueModel(A=fatigue_A, b=fatigue_b)
    wear_model = WearModel()

    total_damage = 0
    proxy_damage = 0
    total_wear = 0
    cycles = 0

    records = []

    while total_damage < 1.0 and cycles < max_cycles:

        t, displacement = blade.simulate(t_end=0.1)
        stress = stress_scale * displacement

        rms = MonitoringFeatures.compute_rms(displacement)
        proxy_damage += rms ** fatigue_b
        peak = MonitoringFeatures.compute_peak(displacement)

        damage = fatigue.compute_damage(stress)
        wear = wear_model.update_wear(stress, cutting_speed=5)

        total_damage += damage
        

        total_wear += wear

        blade.F0 *= (1 + total_wear * 0.0001)

        records.append({
            "cycle": cycles,
            "rms": rms,
            "peak": peak,
            "damage": total_damage,
            "wear": total_wear,
            "proxy_damage": proxy_damage
        })


        cycles += 1

    lifetime = cycles

    df = pd.DataFrame(records)
    df["RUL"] = lifetime - df["cycle"]
    df["damping"] = damping
    df["force_amp"] = force_amp
    df["run_id"] = f"d{damping}_f{force_amp}"


    return lifetime, df
