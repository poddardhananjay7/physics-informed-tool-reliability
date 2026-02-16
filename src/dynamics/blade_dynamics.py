import numpy as np
from scipy.integrate import solve_ivp


class BladeDynamics:
    def __init__(self, mass, damping, stiffness, force_amp, omega):
        self.m = mass
        self.c = damping
        self.k = stiffness
        self.F0 = force_amp
        self.omega = omega

    def cutting_force(self, t):
        noise = 0.05 * self.F0 * np.random.randn()
        return self.F0 * np.sin(self.omega * t) + noise

    def equation(self, t, y):
        x, v = y
        F = self.cutting_force(t)
        dxdt = v
        dvdt = (F - self.c * v - self.k * x) / self.m
        return [dxdt, dvdt]

    def simulate(self, t_end=1.0, n_steps=2000):
        t_eval = np.linspace(0, t_end, n_steps)
        sol = solve_ivp(self.equation, [0, t_end], [0.0, 0.0], t_eval=t_eval)
        return t_eval, sol.y[0]
