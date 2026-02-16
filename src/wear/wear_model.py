import numpy as np


class WearModel:
    def __init__(self, wear_coeff=1e-6):
        self.k_w = wear_coeff

    def update_wear(self, stress_signal, cutting_speed):
        sigma_rms = np.sqrt(np.mean(stress_signal**2))
        wear_increment = self.k_w * sigma_rms * cutting_speed
        return wear_increment
