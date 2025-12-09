import numpy as np


class Optimize:

    def __init__(self, server_freq=[], freq_res=1):
        self.server_freq = server_freq
        self.res_freq = np.linspace(0.1, 1, num=freq_res)

    def cvx_energy(self, task, mec_idx, trans_time, running_time, waiting_time):
        delta_max = 0
        frequencies = self.res_freq * self.server_freq[mec_idx]
        other_time = trans_time + running_time + waiting_time

        for f in frequencies:
            if other_time + task['circle'] / f < delta_max:
                return f
        return np.min(self.res_freq)
