import numpy as np


class FatigueModel:
    def __init__(self, A=5e11, b=3):
        self.A = A
        self.b = b


    def cycles_to_failure(self, stress_amplitude):
        return self.A * (stress_amplitude ** (-self.b))

    def compute_damage(self, stress_signal):
        sigma_a = np.max(np.abs(stress_signal))
        N = self.cycles_to_failure(sigma_a)
        damage = 1.0 / N
        return damage
