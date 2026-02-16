import numpy as np


class MonitoringFeatures:

    @staticmethod
    def compute_rms(signal):
        return np.sqrt(np.mean(signal**2))

    @staticmethod
    def compute_peak(signal):
        return np.max(np.abs(signal))

    @staticmethod
    def dominant_frequency(signal, sampling_rate):
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)
        idx = np.argmax(np.abs(fft_vals))
        return abs(freqs[idx])
